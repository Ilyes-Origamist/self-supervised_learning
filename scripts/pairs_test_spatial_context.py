"""Diagnostic for the spatial context task: does the trained network use the relation between the two patches?

On test images, measures the pretext accuracy (8 relative positions, chance 12.5%) of a checkpoint when:
  normal      : both patches come from the same image (the real task)
  swap-first  : the first patch comes from another image (same positions, same label)
  swap-second : the second patch comes from another image
If swap-first stays close to normal, the head ignores the first patch: the task has degenerated into
predicting the second patch's absolute position in the image (up to ~29% with this sampling).
If both swaps drop towards 12.5%, the network does reason about the pair.

Also checks that the siamese implementation (both patches in one batch through one trunk) gives the
same outputs as two separate branches with shared weights, as drawn in the paper (Fig. 3).

Usage (from cs294-158-ssl/, with no training running on the GPU):
    python pairs_test_spatial_context.py [-d tiny] [-n 3]
Read-only: loads the checkpoint and the test set, writes nothing.
"""
import argparse
import os.path as osp

import torch

import sys
# repository root, so that deepul_helper is found without `pip install -e .`
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from deepul_helper.data import SPATIAL_CONTEXT, get_datasets
from deepul_helper.tasks import SpatialContext
from deepul_helper.utils import remove_module_state_dict
from probe_spatial_context import load_split, batches

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='tiny', help='tiny|cifar10|stl10|imagenette (default: tiny)')
parser.add_argument('-c', '--ckpt', type=str, default=None,
                    help='default: results/<dataset>_spatial_context/checkpoint.pth.tar')
parser.add_argument('-n', '--repeats', type=int, default=3,
                    help='passes over the test set (new random pairs each time, default: 3)')
parser.add_argument('-b', '--batch_size', type=int, default=256, help='default: 256')
parser.add_argument('-s', '--seed', type=int, default=0, help='default: 0')


def pretext_logits(model, patches1, patches2):
    """Same computation as SpatialContext.forward: both patches in one batch through the shared trunk"""
    n = patches1.shape[0]
    feats = model.model(torch.cat((patches1, patches2), dim=0), out_feat_keys=('conv4',)).mean(dim=(2, 3))
    return model.head(torch.cat((feats[:n], feats[n:]), dim=1))


def pretext_logits_two_branches(model, patches1, patches2):
    """Paper-style siamese network: one pass per patch through the same (shared) trunk"""
    f1 = model.model(patches1, out_feat_keys=('conv4',)).mean(dim=(2, 3))
    f2 = model.model(patches2, out_feat_keys=('conv4',)).mean(dim=(2, 3))
    return model.head(torch.cat((f1, f2), dim=1))


def other_image(patches):
    # patch i is replaced by the patch of image i-1 (a different image: the test set is shuffled)
    return patches.roll(1, dims=0)


@torch.no_grad()
def run(model, x, y, args):
    correct = {'normal': 0, 'swap-first': 0, 'swap-second': 0}
    max_diff, total = 0., 0
    for _ in range(args.repeats):
        for xb, _ in batches(x, y, args.batch_size, shuffle=True):
            patches1, patches2, targets = model._get_patches(xb)
            logits = pretext_logits(model, patches1, patches2)
            logits_branches = pretext_logits_two_branches(model, patches1, patches2)
            max_diff = max(max_diff, (logits - logits_branches).abs().max().item())

            correct['normal'] += logits.argmax(dim=-1).eq(targets).sum().item()
            correct['swap-first'] += pretext_logits(model, other_image(patches1), patches2) \
                .argmax(dim=-1).eq(targets).sum().item()
            correct['swap-second'] += pretext_logits(model, patches1, other_image(patches2)) \
                .argmax(dim=-1).eq(targets).sum().item()
            total += targets.shape[0]
    return {k: 100. * v / total for k, v in correct.items()}, max_diff, total


if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.ckpt is None:
        args.ckpt = osp.join('results', f'{args.dataset}_spatial_context', 'checkpoint.pth.tar')
    if not osp.exists(args.ckpt):
        raise SystemExit(f'No checkpoint at {args.ckpt}. Train spatial_context on {args.dataset} first '
                         f'or pass one with -c.')

    size = SPATIAL_CONTEXT[args.dataset]['size']  # the view the patches are cut from
    print(f'Preparing the {size}x{size} test views...', flush=True)
    _, test_dset, n_classes = get_datasets(args.dataset, 'spatial_context')
    xte, yte = load_split(test_dset, 'test', [size], size)

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model = SpatialContext(args.dataset, n_classes)
    model.load_state_dict(remove_module_state_dict(ckpt['state_dict']))
    model.cuda().eval()
    print(f'Checkpoint {args.ckpt}: epoch {ckpt["epoch"]} | patch {model.patch_size}, gap {model.gap}, '
          f'jitter {model.jitter}, colour mode {model.color_mode}', flush=True)

    acc, max_diff, total = run(model, xte[size], yte, args)
    print(f'\nSiamese check: one batch through the shared trunk vs two separate branches, '
          f'max |logit difference| = {max_diff:.2e} (float32 rounding, e.g. ~1e-4, is fine)')
    print(f'\nPretext accuracy on {total} test pairs ({args.dataset}, chance 12.50%)')
    for name in ('normal', 'swap-first', 'swap-second'):
        print(f'  {name:<12} {acc[name]:6.2f}%')
