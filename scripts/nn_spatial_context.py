"""Nearest neighbours of a 'cat' (or 'car') image in the feature space of the spatial context network.

Same idea as the nearest-neighbour demo (deepul_helper/demos.py, display_nearest_neighbors): every test
image is encoded with the trained network, and we show the k test images whose features are the closest
(Euclidean distance) to the features of a query image. Here, the query is a random test image of the cat
class (QUERY_CLASS below; the car example is kept, commented), for each dataset. The features are the ones
used by the linear classifier: conv2 of the 32x32 image.

Each neighbour has a green frame if it belongs to the same class as the query, red otherwise, and the
script prints how many of the k neighbours belong to that class.

Usage (from cs294-158-ssl/):
    python scripts/nn_spatial_context.py [-d cifar10,tiny,stl10] [-c best|last] [-k 16] [-s 0]
Output: one figure per dataset, nn_spatial_context_<dataset>_<class>.png (in --out, default results_nn/)
"""
import argparse
import os
import os.path as osp
import sys

# repository root, so that deepul_helper is found without `pip install -e .`
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data

from deepul_helper.data import SPATIAL_CONTEXT
from deepul_helper.demos import load_model_and_data

# Class of the query image: its label in the files used for training, and its name.
# STL-10 was downloaded from the Hugging Face dataset jxie/stl10, whose labels are the texts
# '1', '10', '2', ..., '9' sorted alphabetically: label 3 is the original class 3 ('car'), label 4 the
# original class 4 ('cat'). Tiny ImageNet labels are those of the Hugging Face dataset zh-plus/tiny-imagenet.

# Car example (used first):
# QUERY_TAG = 'car'
# QUERY_CLASS = {
#     'cifar10': (1, 'automobile'),
#     'stl10': (3, 'car'),
#     # Tiny ImageNet has no plain 'car' class; we use 'sports car' (n04285008)
#     # (other car classes: convertible 86, beach wagon 64)
#     'tiny': (142, 'sports car'),
# }

# Cat example:
QUERY_TAG = 'cat'
QUERY_CLASS = {
    'cifar10': (3, 'cat'),
    'stl10': (4, 'cat'),
    # Tiny ImageNet has no plain 'cat' class; we use 'tabby cat' (n02123045)
    # (other cat classes: Persian cat 28, Egyptian cat 29)
    'tiny': (27, 'tabby cat'),
}

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--datasets', type=str, default='cifar10,tiny,stl10',
                    help='comma-separated list (default: cifar10,tiny,stl10)')
parser.add_argument('-c', '--ckpt', type=str, default='best', choices=('best', 'last'),
                    help='best = model_best.pth.tar (lowest pretext loss), last = checkpoint.pth.tar (default: best)')
parser.add_argument('-k', type=int, default=16, help='number of neighbours (default: 16)')
parser.add_argument('-s', '--seed', type=int, default=0, help='seed used to pick the query image (default: 0)')
parser.add_argument('-o', '--out', type=str, default='results_nn', help='output folder (default: results_nn)')


@torch.no_grad()
def encode_test_set(model, test_dset, batch_size=256):
    """Features (conv2 of the 32x32 image, as for the linear classifier), labels and 32x32 images."""
    # shuffle=False: the images keep the order of the dataset, so the same seed gives the same query
    loader = data.DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=2)
    feats, labels, images = [], [], []
    for (_, x32), y in loader:  # each sample gives (patch view, 32x32 view): we only need the 32x32 one
        x = x32.cuda()
        if model.color_mode == 'project':  # same colour pre-processing as during training
            x = model._project_colors(x)
        feats.append(model.encode(x).cpu())
        labels.append(y)
        images.append(x32)
    return torch.cat(feats), torch.cat(labels), torch.cat(images)


def to_displayable(images, dataset):
    """Undo the normalisation of the images, for display (values between 0 and 1)."""
    mean = torch.tensor(SPATIAL_CONTEXT[dataset]['mean']).view(1, 3, 1, 1)
    std = torch.tensor(SPATIAL_CONTEXT[dataset]['std']).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()


def plot(query_img, nn_imgs, same_class, dataset, class_name, k, path):
    cols = 4
    rows = int(np.ceil(k / cols))
    fig = plt.figure(figsize=(2 + cols * 1.2, rows * 1.2))
    grid = fig.add_gridspec(rows, cols + 2)

    ax = fig.add_subplot(grid[:, :2])  # the query, on the left
    ax.imshow(query_img, interpolation='nearest')
    ax.set_title(f'Query ({class_name})', fontsize=9)
    ax.axis('off')

    for i in range(k):  # the neighbours, closest first, with a green / red frame
        ax = fig.add_subplot(grid[i // cols, 2 + i % cols])
        ax.imshow(nn_imgs[i], interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('tab:green' if same_class[i] else 'tab:red')
            spine.set_linewidth(2.5)
    fig.suptitle(f'Spatial context, {dataset}: {int(same_class.sum())}/{k} nearest neighbours are '
                 f'{class_name}', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':  # needed on Windows, where the data loading workers re-import this file
    args = parser.parse_args()
    ckpt_name = {'best': 'model_best.pth.tar', 'last': 'checkpoint.pth.tar'}[args.ckpt]
    os.makedirs(args.out, exist_ok=True)

    for dataset in args.datasets.split(','):
        if not osp.exists(osp.join('results', f'{dataset}_spatial_context', ckpt_name)):
            print(f'No spatial context model for {dataset}, skipping.')
            continue
        query_label, class_name = QUERY_CLASS[dataset]

        model, _, _, test_loader = load_model_and_data('spatial_context', dataset, ckpt_name=ckpt_name,
                                                       with_train_loader=False)
        feats, labels, images = encode_test_set(model, test_loader.dataset)

        # a random test image of the chosen class is the query
        class_idx = torch.nonzero(labels == query_label).flatten().numpy()
        q = np.random.RandomState(args.seed).choice(class_idx)

        # Euclidean distance between the query and every test image; the query itself is excluded
        dists = torch.cdist(feats[q:q + 1], feats).flatten()
        dists[q] = float('inf')
        nn_idx = torch.topk(dists, args.k, largest=False).indices

        same_class = (labels[nn_idx] == query_label).numpy()
        print(f'{dataset}: query = test image {q} ({class_name}); {int(same_class.sum())}/{args.k} nearest '
              f'neighbours are {class_name} (chance: {100 * len(class_idx) / len(labels):.1f} % of the test set)')

        path = osp.join(args.out, f'nn_spatial_context_{dataset}_{QUERY_TAG}.png')
        plot(to_displayable(images[q:q + 1], dataset)[0], to_displayable(images[nn_idx], dataset),
             same_class, dataset, class_name, args.k, path)
        print(f'  saved {path}')

        del model, test_loader, feats, images
        torch.cuda.empty_cache()
