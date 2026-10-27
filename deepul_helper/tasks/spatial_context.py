import torch
import torch.nn as nn
import torch.nn.functional as F

from .rotation import NetworkInNetwork

"""
Implements the spatial context task from the paper https://arxiv.org/pdf/1505.05192.pdf
But uses a different architecture (Network in Network, same trunk as RotationPrediction)
instead of AlexNet, scaled down to 64x64 images:
    11x11 patches (17% of the image side; paper: 96 px on ~390-670 px images, i.e. 14-25%),
    gap of 5 (~half a patch, as in the paper), jitter of +-1 per patch.
    As in the paper (Sec. 3), the first patch is sampled anywhere in the image, not always at the centre:
    with a fixed centre anchor, the label is the second patch's absolute position, a trivial shortcut
    (position alone gives 100% accuracy; 27% here, chance 12.5%).
Siamese late-fusion structure as in the paper: shared trunk per patch, small fused head -> 8 positions.
Colour handling against chromatic aberration (paper, Sec. 3.1), selected by `color_mode`:
    'drop'    : colour dropping, each patch keeps one random colour channel, the two others are
                replaced by Gaussian noise (std = 1/100 of the kept channel's std); patches only
    'project' : colour projection, every pixel of the mean-subtracted RGB image is multiplied by
                B = I - a^T a / (a a^T), a = [-1, 2, -1] (green-magenta axis); applied to the patches
                and to the probe images, so that the trunk always sees projected colours
    None      : no colour handling
The linear probe (trained online, as for the other tasks) reads conv2 of the full 32x32 image,
exactly like RotationPrediction, so the two tasks only differ by their pretext task.
"""
class SpatialContext(nn.Module):
    metrics = ['Loss', 'Acc1']
    metrics_fmt = [':.4e', ':6.2f']

    def __init__(self, dataset, n_classes):
        super().__init__()
        self.patch_size = 11 # 11x11 patches on a 64x64 image
        self.gap = 5 # around half the patch size, as in the paper
        self.jitter = 1 # jitter of +-1 pixel per patch (keeps the gap between 3 and 7 pixels)
        self.color_mode = 'drop' # 'drop' | 'project' | None, against chromatic aberration (paper, Sec. 3.1)
        assert self.color_mode in ('drop', 'project', None), self.color_mode
        if dataset in ('cifar10', 'tiny'):
            self.model = NetworkInNetwork()  # fully convolutional: 11x11 patch -> conv4 192x3x3
            # NiN's own classifier block (GAP -> 4 rotation classes) is never used: the 8 positions
            # depend on both patches, so they are predicted by the fused head below
            self.model._feature_blocks[-1] = nn.Identity()
            self.latent_dim = 192 * 8 * 8    # conv2 of a full 32x32 image, as in RotationPrediction
            self.feat_layer = 'conv2'
        else:
            raise Exception('Unsupported dataset:', dataset)

        # Late fusion ("fc7 -> fc8 -> fc9" in the paper): only these layers see both patches
        self.head = nn.Sequential(
            nn.Linear(2 * 192, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 8)
        )
        self.dataset = dataset
        self.n_classes = n_classes

        # Colour projection on normalised images: undo the per-channel std of data.py's Normalize
        # (the mean is already subtracted), project, redo it: x' = diag(1/std) B diag(std) x
        std = {'cifar10': (0.247, 0.243, 0.261), 'tiny': (0.2770, 0.2691, 0.2821)}[dataset]
        a = torch.tensor([[-1., 2., -1.]])
        B = torch.eye(3) - a.t() @ a / (a @ a.t())
        std = torch.tensor(std)
        # not persistent: derived from constants, keeps checkpoints compatible across colour modes
        self.register_buffer('color_projection', B * std[None, :] / std[:, None], persistent=False)

    def construct_classifier(self):
        return nn.Sequential(
            nn.BatchNorm1d(self.latent_dim, affine=False),
            nn.Linear(self.latent_dim, self.n_classes)
        )

    def forward(self, images):
        # 64x64 view for the patches, 32x32 view of the same image for the linear probe
        images, images_probe = images
        batch_size = images.shape[0]
        patches1, patches2, targets = self._get_patches(images)

        # Siamese trunk with tied weights: both patches in one batch, then split
        feats = self.model(torch.cat((patches1, patches2), dim=0), out_feat_keys=('conv4',))
        feats = feats.mean(dim=(2, 3))  # global average pooling -> (2B, 192)
        logits = self.head(torch.cat((feats[:batch_size], feats[batch_size:]), dim=1))
        loss = F.cross_entropy(logits, targets)

        pred = logits.argmax(dim=-1)
        correct = pred.eq(targets).float().sum()
        acc = correct / targets.shape[0] * 100.

        return dict(Loss=loss, Acc1=acc), self._probe_features(images_probe)

    def encode(self, images):
        zs = self.model(images, out_feat_keys=(self.feat_layer,))
        return zs.flatten(start_dim=1)

    def _probe_features(self, images):
        # The trunk's BatchNorm statistics come from patches. Use the running statistics
        # (eval mode) for full images in training too, so that the probe is trained on the
        # same features it is evaluated on. No gradient: the probe input is detached anyway.
        training = self.model.training
        self.model.eval()
        with torch.no_grad():
            if self.color_mode == 'project':
                images = self._project_colors(images)
            zs = self.encode(images)
        self.model.train(training)
        return zs

    def _get_patches(self, images):
        batch_size, dev = images.shape[0], images.device
        P, J = self.patch_size, self.jitter # 11, 1
        size = images.shape[-1]  # 64
        stride = P + self.gap    # 16: offset between the two patches along each axis
        # steps (dy, dx) for labels 0..7 = paper's positions 1..8
        steps = torch.tensor([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
                             device=dev)
        # Label first (uniform, so the 8 classes stay balanced), then the first patch uniformly among
        # the positions where that neighbour fits in the image: equivalent to the paper's "first patch
        # uniform, then one of 8 neighbours" on large images, where almost every neighbour fits
        targets = torch.randint(8, (batch_size,), device=dev)

        def anchor(step):
            # step: along x or y, in {-1, 0, 1}, for each sample in the batch
            # top-left coordinate of the first patch along one axis, so that both patches stay inside
            # the image after jitter: [J, size - P - J], shifted by one stride on the neighbour's side
            lo = J + (step < 0).long() * stride
            hi = (size - P) - J - (step > 0).long() * stride
            return lo + (torch.rand(batch_size, device=dev) * (hi - lo + 1)).long()

        def jitter(pos):
            # each patch is jittered independently, as in the paper
            pos = pos + torch.randint(-J, J + 1, (batch_size,), device=dev)
            return pos.clamp(0, size - P)

        dy, dx = steps[targets, 0], steps[targets, 1]
        ay, ax = anchor(dy), anchor(dx)
        y1, x1 = jitter(ay), jitter(ax)                            # first patch, anywhere
        y2, x2 = jitter(ay + dy * stride), jitter(ax + dx * stride)  # its neighbour in direction `target`

        def crop(x, y0, x0):
            ar = torch.arange(P, device=dev)
            b = torch.arange(x.shape[0], device=dev)[:, None, None]
            rows, cols = (y0[:, None] + ar)[:, :, None], (x0[:, None] + ar)[:, None, :]
            return x[b, :, rows, cols].permute(0, 3, 1, 2)  # (B, P, P, C) -> (B, C, P, P)

        patches1, patches2 = crop(images, y1, x1), crop(images, y2, x2)
        # applied in train and eval mode: it is part of the pretext task's input
        if self.color_mode == 'drop':
            patches1, patches2 = self._drop_colors(patches1), self._drop_colors(patches2)
        elif self.color_mode == 'project':
            patches1, patches2 = self._project_colors(patches1), self._project_colors(patches2)
        return patches1, patches2, targets

    def _project_colors(self, images):
        # multiply every pixel's colour vector by the projection matrix (removes the green-magenta axis)
        return torch.einsum('ij,bjhw->bihw', self.color_projection, images)

    def _drop_colors(self, patches):
        # keep one random channel per patch, replace the two others by low-variance Gaussian noise
        keep = torch.randint(3, (patches.shape[0],), device=patches.device)
        mask = F.one_hot(keep, 3).to(patches.dtype)[:, :, None, None]           # (B, 3, 1, 1)
        kept_std = (patches * mask).sum(dim=1).flatten(start_dim=1).std(dim=1)  # (B,)
        noise = torch.randn_like(patches) * kept_std[:, None, None, None] / 100.
        return patches * mask + noise * (1 - mask)
