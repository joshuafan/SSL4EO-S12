#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import sys
import time
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from models.moco import loader
from models.moco import builder
import pdb

from torch.utils.tensorboard import SummaryWriter

#from datasets.BigEarthNet.bigearthnet_dataset_seco import Bigearthnet
#from datasets.BigEarthNet.bigearthnet_dataset_seco_lmdb import LMDBDataset
from datasets.SSL4EO.ssl4eo_dataset_lmdb import LMDBDataset
#from models.rs_transforms_uint8 import RandomChannelDrop,RandomBrightness,RandomContrast,ToGray

# Try fixing this error https://stackoverflow.com/questions/59809785/i-get-a-attributeerror-module-collections-has-no-attribute-iterable-when-i
import collections
collections.Iterable = collections.abc.Iterable
from cvtorchvision import cvtransforms
#from torchsat.transforms import transforms_cls


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--checkpoints', metavar='DIR', default='./',
                    help='path to checkpoints')
parser.add_argument('--save_path', metavar='DIR', default='./',
                    help='path to save trained model')
parser.add_argument('--bands', type=str, default='B12',
                    help='bands to process')                    
parser.add_argument('--lmdb', action='store_true',
                    help='use lmdb dataset') 
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

parser.add_argument('--normalize', action='store_true', default=False)
parser.add_argument('--subset', action='store_true', default=False)
parser.add_argument('--mode', nargs='*', default=['s2c'])
parser.add_argument('--dtype', type=str, default='uint8')
parser.add_argument('--season', type=str, default='augment')

parser.add_argument('--in_size', type=int, default=224)
parser.add_argument("--is_slurm_job", action='store_true', help="running in slurm")

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, season='fixed'):
        self.base_transform = base_transform
        self.season = season

    def __call__(self, x):

        if self.season=='augment':
            season1 = np.random.choice([0,1,2,3])
            season2 = np.random.choice([0,1,2,3])
        elif self.season=='fixed':
            np.random.seed(42)
            season1 = np.random.choice([0,1,2,3])
            season2 = season1
        elif self.season=='random':
            season1 = np.random.choice([0,1,2,3])
            season2 = season1

        x1 = np.transpose(x[season1,:,:,:],(1,2,0))
        x2 = np.transpose(x[season2,:,:,:],(1,2,0))

        q = self.base_transform(x1)
        k = self.base_transform(x2)

        return [q, k]


def main():
    print("Main pretrain")

    args = parser.parse_args()

    '''
    if args.rank==0 and not os.path.isdir(args.checkpoints):
        os.makedirs(args.checkpoints,exist_ok=True)
    if args.rank==0:
        tb_writer = SummaryWriter(os.path.join(args.checkpoints,'log'))
    '''

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        '''
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        '''
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    # @joshuafan temporarily removed
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])


    ### add slurm option ###
    args.is_slurm_job = "SLURM_JOB_ID" in os.environ
    if args.is_slurm_job:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NNODES"]) * int(
            os.environ["SLURM_TASKS_PER_NODE"][0]
        )


    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    # @joshuafan added: If not using distributed mode (only 1 GPU), set "args.gpu" to the GPU that we have
    print("Num gpus:", ngpus_per_node, "CUDA_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"])
    if not args.distributed:
        gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        assert len(gpus) == 1, f"There are multiple GPUs {gpus}, but args.distributed is False!"
        args.gpu = int(gpus[0])  #f"cuda:{gpus[0]}"

    # Otherwise if using distributed, delete the communication file first if it exists
    if os.path.isfile(args.dist_url.removeprefix('file://')):
        os.remove(args.dist_url.removeprefix('file://'))
        print("File existed", args.dist_url.removeprefix('file://'))

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu_num, ngpus_per_node, args):
    gpu = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[gpu_num])  #  gpu  @joshuafan changed
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0 or (args.is_slurm_job and args.rank != 0):
        print("Got inside the if statement", gpu, args.rank)
        # def print_pass(*args):
        #     pass
        # builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        sys.stdout.flush()

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu

        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    print("gpu_num", gpu_num, "gpu", args.gpu, "cudavisibledevices",  os.environ["CUDA_VISIBLE_DEVICES"], "args.rank", args.rank)

    # create tb_writer
    if args.rank==0 and not os.path.isdir(args.checkpoints):
        os.makedirs(args.checkpoints, exist_ok=True)
    if args.rank==0:
        tb_writer = SummaryWriter(os.path.join(args.checkpoints,'log'))    

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = builder.MoCo(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, bands=args.bands, distributed=args.distributed)
    
    print('model created.')
    sys.stdout.flush()

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        print("Distributed")

        ### add slurm option ###
        if args.is_slurm_job:
            args.gpu_to_work_on = args.rank % torch.cuda.device_count()
            print("Slurm job", args.gpu_to_work_on)
            sys.stdout.flush()
            torch.cuda.set_device(args.gpu_to_work_on)
            print("torch cuda set device", args.gpu_to_work_on)
            sys.stdout.flush()
            model.cuda()
            print("model.cuda() finished")
            sys.stdout.flush()
            model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu_to_work_on], output_device=[args.gpu_to_work_on])   
            print('model distributed.')
            sys.stdout.flush()     
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            print('model distributed but slurm job was not set')
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            print('distributed, but not slurm job and gpu is none')
    elif args.gpu is not None:
        print('not distributed, gpu', args.gpu)
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    ### load dataset
    
    lmdb = args.lmdb
    
    if args.bands == 'B13':
        n_channels = 13
    elif args.bands == 'B12':
        n_channels = 12

    # If also using Sentinel-1, add 2 bands
    if args.mode == ["s1", "s2c"]:
        n_channels += 2
    else:
        assert args.mode == ["s2c"], "We assume the mode is just `s2c`"

    # if args.dtype=='uint8':
    #     from models.rs_transforms_uint8 import RandomChannelDrop,RandomBrightness,RandomContrast,ToGray
    # else:
    #     from models.rs_transforms_float32 import RandomChannelDrop,RandomBrightness,RandomContrast,ToGray
    # @joshuafan even with uint8 data, we now convert to float32 before transforms. This is because Kornia
    # transforms can only be used with float data.
    from models.rs_transforms_float32 import RandomChannelDrop,RandomBrightness,RandomContrast,ToGray

    train_transforms = cvtransforms.Compose([
        #cvtransforms.Resize(128),
        cvtransforms.RandomResizedCrop(args.in_size, scale=(0.2, 1.)),
        cvtransforms.RandomApply([
            RandomBrightness(0.4),
            RandomContrast(0.4)
        ], p=0.8),
        cvtransforms.RandomApply([ToGray(n_channels)], p=0.2),
        cvtransforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
        cvtransforms.RandomHorizontalFlip(),       
        cvtransforms.ToTensor()
        #cvtransforms.RandomApply([RandomChannelDrop(min_n_drop=1, max_n_drop=6)], p=0.5),        
        ])
    

    train_dataset = LMDBDataset(
        lmdb_file=args.data,
        s1_transform=TwoCropsTransform(train_transforms,season=args.season),
        s2c_transform=TwoCropsTransform(train_transforms,season=args.season),
        is_slurm_job=args.is_slurm_job,
        normalize=args.normalize,
        dtype=args.dtype,
        mode=args.mode,
        subset=args.subset
    )   
    
        
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=args.is_slurm_job, sampler=train_sampler, drop_last=True)

    print('start training...')
    sys.stdout.flush()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        loss,top1,top5 = train(train_loader, model, criterion, optimizer, epoch, args)
        if args.rank==0:    
            tb_writer.add_scalar('loss',loss,global_step=epoch,walltime=None)
            tb_writer.add_scalar('acc1',top1,global_step=epoch,walltime=None)
            tb_writer.add_scalar('acc5',top5,global_step=epoch,walltime=None)


        if epoch%10==9:
            if args.rank==0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(args.checkpoints,'checkpoint_{:04d}.pth.tar'.format(epoch)))
    
    print('Training finished.')
    if args.rank==0:
        tb_writer.close()

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, images in enumerate(train_loader):
        if type(images) is tuple:
            # If using both S1 and S2C, concatenate the images along the channel dimension for now
            images = torch.concatenate(images, dim=1)  # [batch, channel, height, width]
            # print("Concatenated image shape", images.shape)

        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        # print("images shape", images[0].shape, images[1].shape)
        output, target = model(im_q=images[0], im_k=images[1])
        # print("output", output.shape, "target", target.shape)
        # sys.stdout.flush()
        loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    '''
    if args.rank==0:    
        tb_writer.add_scalar('loss',losses.avg,global_step=epoch,walltime=None)
        tb_writer.add_scalar('acc1',top1.avg,global_step=epoch,walltime=None)
        tb_writer.add_scalar('acc5',top5.avg,global_step=epoch,walltime=None)
    '''
    return losses.avg, top1.avg, top5.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

                                                        
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    ss_time = time.time()
    
    # moco-v2
    #args.mlp = True
    #args.moco_t = 0.2
    #args.aug_plus = True
    #args.cos = True
    
    main()
    print('total time: %s.' % (time.time()-ss_time))
