"""Diagnostic for the spatial context task: why is its linear probe near chance?

Trains the same linear probe as SpatialContext.construct_classifier() offline, on frozen features
of full images, for several trunks, layers and input resolutions:
  variants (trunk):
    random   : random-init trunk, BatchNorm statistics computed on full images (baseline)
    patch-bn : checkpoint, BatchNorm statistics from patches (= the online probe's condition)
    image-bn : checkpoint, BatchNorm statistics recomputed on full images at the probe resolution
  layers: conv2 (rotation's probe layer), conv3, conv4 (what the pretext head reads)
  resolutions: 32 (the 32x32 view the online probe uses) or the size of the view the patches are cut
               from (64 for cifar10/tiny, 96 for stl10, 128 for imagenette: the scale the trunk was
               trained at); feature maps are average-pooled to 8x8, so the probe always reads
               192 x 8 x 8 = 12288 features.
  Unlabelled training images (STL-10, label -1) are skipped: the probe trains on the 5k labelled ones.
First run (tiny, conv2, 32x32): random 10.8%, patch-bn 4.6%, image-bn 4.5% test top-1
-> not a BatchNorm issue; this sweep tests the layer and the scale.

Usage (from cs294-158-ssl/, with no training running on the GPU):
    python probe_spatial_context.py -d tiny
    python probe_spatial_context.py -d cifar10 -c results/tiny_spatial_context/checkpoint.pth.tar
      (CIFAR-10 probe of the trunk trained on Tiny ImageNet, until a CIFAR-10 checkpoint exists)
Read-only: loads the checkpoint and the dataset, writes nothing.
Needs about 3 GB of RAM for tiny at both resolutions (use -r 32 to reduce it).
"""
import argparse
import os.path as osp
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from deepul_helper.data import SPATIAL_CONTEXT, get_datasets
from deepul_helper.tasks import SpatialContext
from deepul_helper.utils import accuracy, remove_module_state_dict

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='tiny',
                    help='tiny|cifar10|stl10|imagenette (default: tiny)')
parser.add_argument('-c', '--ckpt', type=str, default=None,
                    help='default: results/<dataset>_spatial_context/checkpoint.pth.tar')
parser.add_argument('-v', '--variants', type=str, default='random,image-bn',
                    help='comma-separated subset of random,patch-bn,image-bn (default: random,image-bn)')
parser.add_argument('-l', '--layers', type=str, default='conv2,conv3,conv4', help='default: conv2,conv3,conv4')
parser.add_argument('-r', '--resolutions', type=str, default=None,
                    help='default: 32 and the patch view size (32,64 for tiny/cifar10, 32,96 stl10, 32,128 imagenette)')
parser.add_argument('-e', '--epochs', type=int, default=5, help='probe epochs per configuration (default: 5)')
parser.add_argument('-b', '--batch_size', type=int, default=256, help='default: 256')
parser.add_argument('--lr', type=float, default=0.01,
                    help='probe learning rate (default: 0.01; the online probe uses 0.1, unstable here)')
parser.add_argument('-s', '--seed', type=int, default=0, help='default: 0')
parser.add_argument('-i', '--log_interval', type=int, default=100, help='default: 100')

torch.backends.cudnn.benchmark = True
FEAT_SIZE = 8  # feature maps are pooled to 8x8, the conv2 size of a 32x32 image


def load_split(dset, name, resolutions, patch_view):
    """Apply the task's transform once (same preprocessing as training) and keep the requested views
    of the labelled images (STL-10's unlabelled training images, label -1, are skipped)"""
    views = {patch_view: 0, 32: 1}  # TwoResTransform returns (patch view, 32x32 view)
    labels = getattr(dset, 'y', None)  # known in advance for MemmapImages / ArrayImages
    indices = range(len(dset)) if labels is None else [i for i in range(len(dset)) if labels[i] >= 0]
    xs = {r: [] for r in resolutions}
    ys = []
    start = time.time()
    for n, i in enumerate(indices):
        imgs, y = dset[i]
        if y < 0:
            continue
        for r in resolutions:
            xs[r].append(imgs[views[r]].half())
        ys.append(y)
        if (n + 1) % 10000 == 0:
            print(f'  {name}: {n + 1}/{len(indices)} images ({time.time() - start:.0f}s)', flush=True)
    return {r: torch.stack(xs[r]) for r in resolutions}, torch.tensor(ys)


def batches(x, y, batch_size, shuffle):
    idx = torch.randperm(len(x)) if shuffle else torch.arange(len(x))
    for i in range(0, len(x), batch_size):
        j = idx[i:i + batch_size]
        yield x[j].cuda().float(), y[j].cuda()


def trunk_input(model, images):
    # same colour preprocessing as SpatialContext._probe_features (the class's current color_mode)
    return model._project_colors(images) if model.color_mode == 'project' else images


@torch.no_grad()
def recalibrate_bn(model, x, y, batch_size):
    """Recompute all trunk BatchNorm statistics on full images: exact average over one pass"""
    bns = [m for m in model.model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative moving average
    model.model.train()
    for xb, _ in batches(x, y, batch_size, shuffle=False):
        model.model(trunk_input(model, xb), out_feat_keys=('conv4',))
    for m in bns:
        m.momentum = 0.1
    model.model.eval()


@torch.no_grad()
def extract(model, x, y, layer, batch_size):
    """Frozen trunk features at `layer`, average-pooled to 8x8 and flattened, cached on CPU in fp16"""
    feats = []
    for xb, _ in batches(x, y, batch_size, shuffle=False):
        f = model.model(trunk_input(model, xb), out_feat_keys=(layer,))
        feats.append(F.adaptive_avg_pool2d(f, FEAT_SIZE).flatten(start_dim=1).half().cpu())
    return torch.cat(feats)


def feature_stats(z):
    """Share of exactly-zero features, and channels that are zero on every test image"""
    chan_max = z.view(len(z), -1, FEAT_SIZE * FEAT_SIZE).float().amax(dim=(0, 2))
    return 100. * (z == 0).float().mean().item(), int((chan_max == 0).sum()), chan_max.numel()


@torch.no_grad()
def evaluate(probe, z, y, batch_size):
    probe.eval()
    correct1, correct5 = 0., 0.
    for zb, yb in batches(z, y, batch_size, shuffle=False):
        acc1, acc5 = accuracy(probe(zb), yb, topk=(1, 5))
        correct1 += acc1.item() * yb.shape[0]
        correct5 += acc5.item() * yb.shape[0]
    return correct1 / len(z), correct5 / len(z)


def train_probe(model, ztr, ytr, zte, yte, args):
    """Same probe and optimizer as the online one (SGD, Nesterov, cosine), on frozen cached features"""
    probe = model.construct_classifier().cuda()
    optimizer = torch.optim.SGD(probe.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    n_iters = (len(ztr) + args.batch_size - 1) // args.batch_size

    for epoch in range(args.epochs):
        probe.train()
        seen, correct1, loss_sum = 0, 0., 0.
        for i, (zb, yb) in enumerate(batches(ztr, ytr, args.batch_size, shuffle=True)):
            logits = probe(zb)
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            seen += yb.shape[0]
            loss_sum += loss.item() * yb.shape[0]
            correct1 += accuracy(logits, yb, topk=(1,))[0].item() * yb.shape[0]
            if i % args.log_interval == 0:
                print(f'    epoch {epoch} [{i:3d}/{n_iters}]  loss {loss_sum / seen:.3f}  '
                      f'train top1 {correct1 / seen:5.2f}', flush=True)
        scheduler.step()

        top1, top5 = evaluate(probe, zte, yte, args.batch_size)
        print(f'    epoch {epoch}: train top1 {correct1 / seen:5.2f} | test top1 {top1:5.2f}  top5 {top5:5.2f}',
              flush=True)
    return correct1 / seen, top1, top5


def build_model(variant, n_classes, args):
    torch.manual_seed(args.seed)
    model = SpatialContext(args.dataset, n_classes)
    if variant != 'random':
        # the probe is not in the checkpoint, so a trunk trained on another dataset loads too
        ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(remove_module_state_dict(ckpt['state_dict']))
    return model.cuda().eval()


if __name__ == '__main__':
    args = parser.parse_args()
    variants = args.variants.split(',')
    layers = args.layers.split(',')
    patch_view = SPATIAL_CONTEXT[args.dataset]['size']
    resolutions = [32, patch_view] if args.resolutions is None else [int(r) for r in args.resolutions.split(',')]
    assert all(v in ('random', 'patch-bn', 'image-bn') for v in variants), variants
    assert all(l in ('conv1', 'conv2', 'conv3', 'conv4') for l in layers), layers
    assert all(r in (32, patch_view) for r in resolutions), f'{resolutions}: use 32 and/or {patch_view}'

    if args.ckpt is None:
        args.ckpt = osp.join('results', f'{args.dataset}_spatial_context', 'checkpoint.pth.tar')
    if any(v != 'random' for v in variants):
        if not osp.exists(args.ckpt):
            raise SystemExit(f'No checkpoint at {args.ckpt}. Train spatial_context on {args.dataset} first, '
                             f'pass another one with -c (e.g. the Tiny ImageNet trunk), or use -v random.')
        ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        print(f'Checkpoint {args.ckpt}: epoch {ckpt["epoch"]}, '
              f'online probe test top1 {float(ckpt["best_acc"]):.2f} (on its own training dataset)', flush=True)
        del ckpt

    print('Preparing the probe views...', flush=True)
    train_dset, test_dset, n_classes = get_datasets(args.dataset, 'spatial_context')
    xtr, ytr = load_split(train_dset, 'train', resolutions, patch_view)
    xte, yte = load_split(test_dset, 'test', resolutions, patch_view)
    print(f'train {len(ytr)} labelled, test {len(yte)} images, {n_classes} classes, '
          f'chance {100. / n_classes:.2f}%', flush=True)

    rows = []
    for variant in variants:
        for res in resolutions:
            print(f'\n=== {variant}, {res}x{res} input ===', flush=True)
            model = build_model(variant, n_classes, args)
            if variant != 'patch-bn':
                print('  recomputing BatchNorm statistics on full images...', flush=True)
                recalibrate_bn(model, xtr[res], ytr, args.batch_size)
            for layer in layers:
                start = time.time()
                ztr = extract(model, xtr[res], ytr, layer, args.batch_size)
                zte = extract(model, xte[res], yte, layer, args.batch_size)
                zero_pct, dead, n_chan = feature_stats(zte)
                print(f'  {layer}: features extracted ({time.time() - start:.0f}s), {zero_pct:.1f}% zeros, '
                      f'{dead}/{n_chan} channels dead on the test set', flush=True)
                train1, top1, top5 = train_probe(model, ztr, ytr, zte, yte, args)
                rows.append((variant, res, layer, zero_pct, dead, train1, top1, top5))
                del ztr, zte
            del model
            torch.cuda.empty_cache()

    print(f'\nOffline linear probe on frozen features of full images '
          f'({args.dataset}, {n_classes} classes, chance {100. / n_classes:.2f}%, '
          f'{args.epochs} epochs, lr {args.lr})')
    print(f'{"variant":<10} {"input":>6} {"layer":>6} {"zeros %":>8} {"dead ch":>8} '
          f'{"train@1":>8} {"test@1":>8} {"test@5":>8}')
    for variant, res, layer, zero_pct, dead, train1, top1, top5 in rows:
        print(f'{variant:<10} {res:>6} {layer:>6} {zero_pct:8.1f} {dead:8d} '
              f'{train1:8.2f} {top1:8.2f} {top5:8.2f}')
