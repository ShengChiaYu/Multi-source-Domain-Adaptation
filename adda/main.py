import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms

import numpy as np
import random
import os
import sys
from os.path import join

from tqdm import tqdm, tnrange
from dataset import dataset_public
from utils import AverageMeter, parse_args
from model import resnet_345, classifier


def load_checkpoint(checkpoint_path, model):
    pretrained_dict = torch.load(checkpoint_path)['state_dict']
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # print(pretrained_dict.keys())
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    print('model loaded from %s' % checkpoint_path)


def fixed_pretrained(model):
    for param in model.parameters():
        param.requires_grad = False


def save_checkpoint(checkpoint_path, CNN, cls, optimizer):
    state = {'CNN_state_dict': CNN.state_dict(),
             'cls_state_dict': cls.state_dict(),
             'optimizer' : optimizer.state_dict(),
             }
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)


def main():
    args = parse_args()

    torch.manual_seed(args.manual_seed)

    dataroot = join(os.getcwd(), args.data_dir)
    transform=transforms.Compose([
              # transforms.RandomCrop(args.img_size, padding=None, pad_if_needed=True, fill=0, padding_mode='edge'),
              transforms.Resize((args.img_size,args.img_size)),
              transforms.ToTensor(),
              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    src_dataset = dataset_public(root=dataroot,
                             transform=transform,
                             train=True,
                             domains=args.source,
    )
    src_loader = torch.utils.data.DataLoader(
        dataset=src_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        )
    tar_dataset = dataset_public(root=dataroot,
                             transform=transform,
                             train=True,
                             domains=args.target,
    )
    tar_loader = torch.utils.data.DataLoader(
        dataset=tar_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        )

    # models
    src_CNN = resnet_345(args).cuda()
    tar_CNN = resnet_345(args).cuda()
    cls = classifier().cuda()

    checkpoint_path = join(os.getcwd(), args.model_dir, args.model_name)
    load_checkpoint(checkpoint_path, src_CNN)
    load_checkpoint(checkpoint_path, tar_CNN)
    load_checkpoint(checkpoint_path, cls)

    fixed_pretrained(src_CNN)
    fixed_pretrained(cls)

    # optimizer
    if args.SGD:
        optimizer = optim.SGD(tar_CNN.parameters(),  lr=args.lr, momentum=0.9)
    elif args.Adam:
        optimizer = optim.Adam(tar_CNN.parameters(), lr=args.lr)

    min_len = min(len(src_loader), len(tar_loader))
    best_acc = 0
    stop_count = 0
    err_log_path = join(os.getcwd(), 'models', 'adda_{}_err.txt'.format(args.target[0]))
    err_log = open(err_log_path, 'w')
    for epoch in range(args.start_epoch, args.max_epochs+1):
        src_CNN.eval()
        tar_CNN.train()
        cls.eval()

        print ('\nEpoch = {}'.format(epoch))
        err_log.write('Epoch = {}, '.format(epoch))

        losses, train_acc = AverageMeter(), AverageMeter()
        train_pbar = tqdm(total=min_len, ncols=100, leave=True)
        for i, (src_data, tar_data) in enumerate(zip(src_loader, tar_loader)):
            src_imgs, _ = src_data
            tar_imgs, tar_labels = tar_data
            # src_imgs, src_labels = src_data

            src_imgs, tar_imgs, tar_labels = src_imgs.cuda(), tar_imgs.cuda(), tar_labels.cuda()
            # src_imgs, tar_imgs, tar_labels, src_labels = src_imgs.cuda(), tar_imgs.cuda(), tar_labels.cuda(), src_labels.cuda()

            tar_CNN.zero_grad()
            src_feature = src_CNN(src_imgs)
            tar_feature = tar_CNN(tar_imgs)

            loss = F.mse_loss(src_feature,tar_feature,reduction='mean')
            loss.backward()

            class_output = cls(tar_feature)
            pred = class_output.max(1, keepdim=True)[1]
            correct = pred.eq(tar_labels.view_as(pred)).sum().item()

            optimizer.step()

            losses.update(loss.data.item(), args.batch_size)
            train_acc.update(correct, args.batch_size)
            train_pbar.update()

            train_pbar.set_postfix({'loss':'{:.4f}'.format(losses.avg),
                                    'acc':'{:.4f}'.format(train_acc.acc),
                                    })
        train_pbar.close()

        if train_acc.acc > best_acc:
            best_acc = train_acc.acc
            stop_count = 0
            checkpoint_path = join(os.getcwd(), 'models', 'adda_{}.pth'.format(args.target[0]))
            save_checkpoint(checkpoint_path, tar_CNN, cls, optimizer)
        else: stop_count += 1

        err_log.write('Loss: {:.4f}, Accuracy: {:.4f}\n'.format(losses.avg, train_acc.acc))
        err_log.flush()

        if stop_count == args.early_stopping: break

    err_log.write('Best test_acc: {:.4f}\n'.format(best_acc))
    err_log.close()

if __name__ == '__main__':
    main()
