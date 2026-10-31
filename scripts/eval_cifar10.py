"""Q3: compare the pretrained self-supervised models on the test set of one dataset.

For each pretext task, we load the trained model and its linear classifier, and measure the top-1 and
top-5 accuracy of the classifier on the test set. The results are printed as one table, and saved as
a CSV file (results_<dataset>.csv) and a bar chart (results_<dataset>.png).

We reuse the functions of the course library (deepul_helper/demos.py), the same ones used in the demos:
    load_model_and_data : loads a model, its linear classifier and the data
    evaluate_classifier : computes the top-1 / top-5 accuracy on a data loader

Usage (from cs294-158-ssl/):
    python scripts/eval_cifar10.py [cifar10|tiny|stl10] [last|best]
    - dataset: cifar10 (default), tiny or stl10
    - checkpoint: last (default) = model of the last epoch,
                  best = model of the epoch with the lowest pretext loss (model_best, as in the demos)
"""
import os.path as osp
import sys

# repository root, so that deepul_helper is found without `pip install -e .`
# sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch

from deepul_helper.demos import load_model_and_data, evaluate_classifier

DATASET = sys.argv[1] if len(sys.argv) > 1 else 'cifar10'  # cifar10 | tiny | stl10
WHICH = sys.argv[2] if len(sys.argv) > 2 else 'last'       # last | best
CKPT_NAME = {'last': 'checkpoint.pth.tar', 'best': 'model_best.pth.tar'}[WHICH]
TASKS = ['context_encoder', 'rotation', 'simclr', 'spatial_context']

torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    rows = []
    for task in TASKS:
        # skip the tasks that were not trained on this dataset
        if not osp.exists(osp.join('results', f'{DATASET}_{task}', CKPT_NAME)):
            print(f'No model for {task} on {DATASET}, skipping.')
            continue

        # only the test set is needed here, so the training images are not loaded
        model, classifier, _, test_loader = load_model_and_data(
            task, DATASET, ckpt_name=CKPT_NAME, with_train_loader=False)
        top1, top5 = evaluate_classifier(model, classifier, test_loader)

        n_params = sum(p.numel() for p in model.parameters()) / 1e6  # size of the model, in millions
        rows.append(dict(Method=task, Params_M=round(n_params, 1),
                         Top1=round(top1, 2), Top5=round(top5, 2)))
        print(rows[-1])

        # free the GPU memory before loading the next model
        del model, classifier, test_loader
        torch.cuda.empty_cache()

    # one table for all the models
    df = pd.DataFrame(rows)
    print(f'\n{DATASET} test accuracy (%), {WHICH} checkpoint')
    print(df.to_string(index=False))
    df.to_csv(f'results_{DATASET}.csv', index=False)

    # and the same results as a bar chart
    ax = df.set_index('Method')[['Top1', 'Top5']].plot.bar(rot=0, figsize=(7, 4))
    ax.set_ylabel('Test accuracy (%)')
    ax.set_title(f'Linear classification on {DATASET} ({WHICH} checkpoint)')
    plt.tight_layout()
    plt.savefig(f'results_{DATASET}.png', dpi=150)
