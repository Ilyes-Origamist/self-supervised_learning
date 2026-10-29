
"""Q3: compare the 3 pretrained self-supervised models on the CIFAR-10 (or Tiny ImageNet) test set. Usage: eval_cifar10.py [cifar10|tiny].
"""
import os.path as osp
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.utils.data as data

from deepul_helper.data import get_datasets
from deepul_helper.tasks import ContextEncoder, RotationPrediction, SimCLR, SpatialContext
from deepul_helper.utils import accuracy, remove_module_state_dict

DATASET = sys.argv[1] if len(sys.argv) > 1 else 'cifar10'  # cifar10 | tiny
TASKS = ['context_encoder', 'rotation', 'simclr', 'spatial_context']
BATCH_SIZE = 128
NUM_WORKERS = 2  # Colab T4 instances only have 2 CPU cores

torch.backends.cudnn.benchmark = True


def to_cuda(images):
    # SimCLR returns a tuple of two views, the other tasks a single tensor
    if isinstance(images, (tuple, list)):
        return [x.cuda(non_blocking=True) for x in images]
    return images.cuda(non_blocking=True)


def load(task):
    """Load model + its linear classifier + the test loader (no train loader needed)."""
    _, test_dset, n_classes = get_datasets(DATASET, task)
    test_loader = data.DataLoader(test_dset, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=True)

    ckpt = torch.load(osp.join('results', f'{DATASET}_{task}', 'model_best.pth.tar'),
                      map_location='cpu', weights_only=False)

    if task == 'context_encoder':
        model = ContextEncoder(DATASET, n_classes)
    elif task == 'rotation':
        model = RotationPrediction(DATASET, n_classes)
    elif task == 'spatial_context':
        model = SpatialContext(DATASET, n_classes)
    else:
        model = SimCLR(DATASET, n_classes, None)
    model.load_state_dict(remove_module_state_dict(ckpt['state_dict']))
    model.cuda().eval()

    classifier = model.construct_classifier()
    classifier.load_state_dict(remove_module_state_dict(ckpt['state_dict_linear']))
    classifier.cuda().eval()
    return model, classifier, test_loader


@torch.no_grad()
def evaluate(model, classifier, loader):
    correct1, correct5 = 0., 0.
    for images, target in loader:
        images, target = to_cuda(images), target.cuda(non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16):  # fp16 on the T4 tensor cores
            _, zs = model(images)
            logits = classifier(zs)
        acc1, acc5 = accuracy(logits.float(), target, topk=(1, 5))
        correct1 += acc1.item() * target.shape[0]
        correct5 += acc5.item() * target.shape[0]
    n = len(loader.dataset)
    return correct1 / n, correct5 / n


if __name__ == '__main__':
    rows = []
    for task in TASKS:
        # if no model for the task for the dataset, skip it
        if not osp.exists(osp.join('results', f'{DATASET}_{task}', 'model_best.pth.tar')):
            print(f'No model for {task} on {DATASET}, skipping.')
            continue
        model, classifier, test_loader = load(task)
        top1, top5 = evaluate(model, classifier, test_loader)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        rows.append(dict(Method=task, Params_M=round(n_params, 1),
                         Top1=round(top1, 2), Top5=round(top5, 2)))
        print(rows[-1])

        # free GPU memory before loading the next model
        del model, classifier, test_loader
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    print(f'\n{DATASET} test accuracy (%)')
    print(df.to_string(index=False))
    df.to_csv(f'results_{DATASET}.csv', index=False)

    ax = df.set_index('Method')[['Top1', 'Top5']].plot.bar(rot=0, figsize=(7, 4))
    ax.set_ylabel('Test accuracy (%)')
    ax.set_title(f'Linear classification on {DATASET}')
    plt.tight_layout()
    plt.savefig(f'results_{DATASET}.png', dpi=150)