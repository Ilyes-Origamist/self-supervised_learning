import argparse
import os
import os.path as osp
import time
import shutil
from warmup_scheduler import GradualWarmupScheduler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim.lr_scheduler as lr_scheduler

from deepul_helper.tasks import *
from deepul_helper.utils import AverageMeter, ProgressMeter, accuracy
from deepul_helper.data import get_datasets
from deepul_helper.lars import LARS


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='cifar10', help='cifar10|tiny|imagenet* (+ stl10 for rotation|spatial_context, imagenette for spatial_context) (default: cifar10)')
parser.add_argument('-t', '--task', type=str, default='rotation',
                    help='context_encoder|rotation|cpc|simclr|spatial_context (default: rotation)')

# Training parameters
parser.add_argument('-b', '--batch_size', type=int, default=128, help='batch size total for all gpus (default: 128)')
parser.add_argument('-e', '--epochs', type=int, default=200, help='default: 200')
parser.add_argument('-o', '--optimizer', type=str, default='sgd', help='sgd|lars|adam (default: sgd)')
parser.add_argument('--lr', type=float, default=0.1, help='default: 0.1')
parser.add_argument('-m', '--momentum', type=float, default=0.9, help='default: 0.9')
parser.add_argument('-w', '--weight_decay', type=float, default=5e-4, help='default: 5e-4')
parser.add_argument('-u', '--warmup_epochs', type=int, default=0,
                    help='# of warmup epochs. If > 0, then the scheduler warmups from lr * batch_size / 256.')

parser.add_argument('-p', '--port', type=int, default=23456, help='tcp port for distributed trainign (default: 23456)')
parser.add_argument('-i', '--log_interval', type=int, default=10, help='default: 10')
parser.add_argument('-r', '--resume', action='store_true',
                    help='resume from results/<dataset>_<task>/checkpoint.pth.tar')


best_loss = float('inf')
best_acc = 0.0

def main():
    args = parser.parse_args()
    assert args.task in ['context_encoder', 'rotation', 'cpc', 'simclr', 'spatial_context']
    assert args.dataset != 'stl10' or args.task in ('rotation', 'spatial_context'), \
        'stl10 is only supported by the rotation and spatial_context tasks'

    args.output_dir = osp.join('results', f"{args.dataset}_{args.task}")
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    ngpus = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, args), join=True)


def main_worker(gpu, ngpus, args):
    global best_loss

    print(f'Starting process on GPU: {gpu}')
    backend = 'nccl' if dist.is_nccl_available() else 'gloo'  # NCCL is Linux-only
    if os.name == 'nt':
        os.environ.setdefault('USE_LIBUV', '0')  # Windows PyTorch builds lack libuv for the TCP store
    dist.init_process_group(backend=backend, init_method=f'tcp://localhost:{args.port}',
                            world_size=ngpus, rank=gpu)
    total_batch_size = args.batch_size
    args.batch_size = args.batch_size // ngpus

    train_dataset, val_dataset, n_classes = get_datasets(args.dataset, args.task)
    # Windows spawns workers, copying the dataset (~0.7 GB for tiny) into each one
    num_workers = min(4 if os.name == 'nt' else 16, os.cpu_count())
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=num_workers, persistent_workers=True,
        pin_memory=True, sampler=train_sampler, drop_last=True
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=num_workers, persistent_workers=True,
        pin_memory=True, drop_last=True, sampler=val_sampler
    )

    if args.task == 'context_encoder':
        model = ContextEncoder(args.dataset, n_classes)
    elif args.task == 'rotation':
        model = RotationPrediction(args.dataset, n_classes)
    elif args.task == 'cpc':
        model = CPC(args.dataset, n_classes)
    elif args.task == 'simclr':
        model = SimCLR(args.dataset, n_classes, dist)
    elif args.task == 'spatial_context':
        model = SpatialContext(args.dataset, n_classes)
    else:
        raise Exception('Invalid task:', args.task)
    args.metrics = model.metrics
    args.metrics_fmt = model.metrics_fmt

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    args.gpu = gpu

    linear_classifier = model.construct_classifier().cuda(gpu)
    if backend == 'gloo' and ngpus == 1:
        # Gloo crashes when all-reducing CUDA tensors on Windows, and one GPU has nothing to sync
        linear_classifier = SingleProcessWrapper(linear_classifier)
        model = SingleProcessWrapper(model)
    else:
        linear_classifier = torch.nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[gpu], find_unused_parameters=True)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=True)
        optimizer_linear = torch.optim.SGD(linear_classifier.parameters(), lr=args.lr,
                                           momentum=args.momentum, nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, 0.999),
                                     weight_decay=args.weight_decay)
        optimizer_linear = torch.optim.Adam(linear_classifier.parameters(), lr=args.lr,
                                            betas=(args.momentum, 0.999))
    elif args.optimizer == 'lars':
        optimizer = LARS(model.parameters(), lr=args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)
        optimizer_linear = LARS(linear_classifier.parameters(), lr=args.lr,
                                momentum=args.momentum)
    else:
        raise Exception('Unsupported optimizer', args.optimizer)

    # Minimize SSL task loss, maximize linear classification accuracy
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, 0, -1)
    scheduler_linear = lr_scheduler.CosineAnnealingLR(optimizer_linear, args.epochs, 0, -1)
    if args.warmup_epochs > 0:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=total_batch_size / 256.,
                                           total_epoch=args.warmup_epochs, after_scheduler=scheduler)
        scheduler_linear = GradualWarmupScheduler(optimizer_linear, multiplier=total_batch_size / 256.,
                                                  total_epoch=args.warmup_epochs,
                                                  after_scheduler=scheduler_linear)

    start_epoch = 0
    if args.resume:
        ckpt_path = osp.join(args.output_dir, 'checkpoint.pth.tar')
        # weights_only=False: the checkpoint stores numpy scalars (best_loss, best_acc)
        ckpt = torch.load(ckpt_path, map_location=f'cuda:{gpu}', weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        linear_classifier.load_state_dict(ckpt['state_dict_linear'])
        optimizer.load_state_dict(ckpt['optimizer'])
        optimizer_linear.load_state_dict(ckpt['optimizer_linear'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scheduler_linear.load_state_dict(ckpt['schedular_linear'])
        best_loss = ckpt['best_loss']
        start_epoch = ckpt['epoch']
        print(f'Resumed from {ckpt_path} at epoch {start_epoch} (best_loss {best_loss:.4f})')

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        train(train_loader, model, linear_classifier,
              optimizer, optimizer_linear, epoch, args)

        val_loss, val_acc = validate(val_loader, model, linear_classifier, args, dist)

        scheduler.step()
        scheduler_linear.step()

        if dist.get_rank() == 0:
            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'state_dict_linear': linear_classifier.state_dict(),
                'optimizer_linear': optimizer_linear.state_dict(),
                'schedular_linear': scheduler_linear.state_dict(),
                'best_loss': best_loss,
                'best_acc': val_acc
            }, is_best, args)


def train(train_loader, model, linear_classifier, optimizer,
          optimizer_linear, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    top1 = AverageMeter('LinearAcc@1', ':6.2f')
    top5 = AverageMeter('LinearAcc@5', ':6.2f')
    avg_meters = {k: AverageMeter(k, fmt)
                  for k, fmt in zip(args.metrics, args.metrics_fmt)}
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, top1, top5] + list(avg_meters.values()),
        prefix="Epoch: [{}]".format(epoch)
    )

    # switch to train mode
    model.train()
    linear_classifier.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute loss
        if isinstance(images, (tuple, list)):
            # Special case for SimCLR (2 augmented views) and spatial_context
            # (pretext view + 32x32 probe view), which return a tuple of 2 image batches
            bs = images[0].shape[0]
            images = [x.cuda(args.gpu, non_blocking=True)
                      for x in images]
        else:
            bs = images.shape[0]
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        out, zs = model(images)
        zs = zs.detach()
        for k, v in out.items():
            avg_meters[k].update(v.item(), bs)

        # compute gradient and optimizer step for ssl task
        optimizer.zero_grad()
        out['Loss'].backward()
        optimizer.step()

        # compute gradient and optimizer step for classifier
        logits = linear_classifier(zs)
        # unlabelled images (label -1, STL-10) only train the pretext task, not the probe
        labelled = target >= 0
        n_labelled = int(labelled.sum())
        if n_labelled > 0:
            loss = F.cross_entropy(logits[labelled], target[labelled])
            acc1, acc5 = accuracy(logits[labelled], target[labelled], topk=(1, 5))
            top1.update(acc1[0], n_labelled)
            top5.update(acc5[0], n_labelled)
        else:
            loss = logits.sum() * 0.  # keeps the backward pass (and DDP's gradient sync) uniform

        optimizer_linear.zero_grad()
        loss.backward()
        optimizer_linear.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0:
            progress.display(i)


def validate(val_loader, model, linear_classifier, args, dist):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    top1 = AverageMeter('LinearAcc@1', ':6.2f')
    top5 = AverageMeter('LinearAcc@5', ':6.2f')
    avg_meters = {k: AverageMeter(k, fmt)
                  for k, fmt in zip(args.metrics, args.metrics_fmt)}
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, top1, top5] + list(avg_meters.values()),
        prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()
    linear_classifier.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # compute and measure loss
            if isinstance(images, (tuple, list)):
                # Special case for SimCLR and spatial_context (tuple of 2 image batches)
                bs = images[0].shape[0]
                images = [x.cuda(args.gpu, non_blocking=True)
                        for x in images]
            else:
                bs = images.shape[0]
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            out, zs = model(images)
            for k, v in out.items():
                avg_meters[k].update(v.item(), bs)

            logits = linear_classifier(zs)
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1.update(acc1[0], bs)
            top5.update(acc5[0], bs)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_interval == 0:
                progress.display(i)

    data = torch.FloatTensor([avg_meters['Loss'].avg, top1.avg, top5.avg] + [v.avg for v in avg_meters.values()])
    data = data.cuda(args.gpu)
    gather_list = [torch.zeros_like(data) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_list, data)
    data = torch.stack(gather_list, dim=0).mean(0).cpu().numpy()

    if dist.get_rank() == 0:
        print_str = f' * LinearAcc@1 {data[1]:.3f} LinearAcc@5 {data[2]:.3f}'
        for i, (k, v) in enumerate(avg_meters.items()):
            print_str += f' {k} {data[i+3]:.3f}'
        print(print_str)

    dist.barrier()
    return data[0], data[1]


class SingleProcessWrapper(nn.Module):
    """Stand-in for DistributedDataParallel with a single process: no collectives,
    same 'module.' prefix in the state_dict, so checkpoints stay compatible"""
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    filename = osp.join(args.output_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, osp.join(args.output_dir, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()
