import torch.optim as optim
import torch.utils.data
import torch.nn as nn
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
from model import resnet_345, inception_345


def fixed_pretrained(cls):
    # fix cnn weight
    for name, child in cls.named_children():
        if name == 'CNN':
            for param in child.parameters():
                param.requires_grad = False


def test(cls, test_loader, args):
    criterion = nn.CrossEntropyLoss()
    cls.eval()
    losses, test_acc = AverageMeter(), AverageMeter()
    test_pbar = tqdm(total=len(test_loader), ncols=100, leave=True)
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.cuda(), labels.cuda()

            if args.inception:
                class_output, _ = cls(images)
            else:
                class_output = cls(images)

            loss = criterion(class_output, labels)
            pred = class_output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct = pred.eq(labels.view_as(pred)).sum().item()

            losses.update(loss.data.item(), args.batch_size)
            test_acc.update(correct, args.batch_size)
            test_pbar.update()

            test_pbar.set_postfix({'loss':'{:.4f}'.format(losses.avg),
                                    'acc':'{:.4f}'.format(test_acc.acc),
                                    })
    test_pbar.close()

    return test_acc, losses


def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict(),
             }
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)


def main():
    args = parse_args()

    torch.manual_seed(args.manual_seed)

    dataroot = join(os.getcwd(), args.data_dir)
    transform=transforms.Compose([
              # transforms.RandomCrop(args.img_size, padding=None, pad_if_needed=True, fill=0, padding_mode='constant'),
              transforms.Resize((args.img_size,args.img_size)),
              transforms.ToTensor(),
              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = dataset_public(root=dataroot,
                             transform=transform,
                             train=True,
                             domains=args.source,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        )
    if args.test:
        transform=transforms.Compose([
                  # transforms.RandomCrop(args.img_size, padding=None, pad_if_needed=True, fill=0, padding_mode='edge'),
                  transforms.Resize((args.img_size,args.img_size)),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        test_dataset = dataset_public(root=dataroot,
                                 transform=transform,
                                 train=False,
                                 domains=args.target,
                                 real_test=args.real_test,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            )

    # models
    if args.inception:
        cls = inception_345().cuda()
    else:
        cls = resnet_345(args).cuda()

    if args.fixed_pretrained:
        fixed_pretrained(cls)

    params_to_update = []
    for name, param in cls.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
    print('Number of parameters to update: {}'.format(len(params_to_update)))

    # optimizer
    if args.SGD:
        optimizer = optim.SGD(params_to_update,  lr=args.lr, momentum=0.9)
    elif args.Adam:
        optimizer = optim.Adam(params_to_update, lr=args.lr)

    if args.resume:
        checkpoint_path = join(os.getcwd(), args.model_dir, args.model_name)
        load_checkpoint(checkpoint_path, cls, optimizer)

    best_acc = 0
    patience = 0
    if len(args.source) > 1:
        err_log_path = join(os.getcwd(), 'models_inception', 'src_combine_tar_{}_err.txt'.format(args.target[0]))
    else:
        err_log_path = join(os.getcwd(), 'models_inception', 'src_{}_err.txt'.format(args.source[0]))

    err_log = open(err_log_path, 'w')
    for epoch in range(args.start_epoch, args.max_epochs+1):
        criterion = nn.CrossEntropyLoss()
        cls.train()

        print ('\nEpoch = {}'.format(epoch))
        err_log.write('Epoch = {}, '.format(epoch))

        losses, train_acc = AverageMeter(), AverageMeter()
        train_pbar = tqdm(total=len(train_loader), ncols=100, leave=True)
        for i, (images, labels) in enumerate(train_loader):

            images, labels = images.cuda(), labels.cuda()

            cls.zero_grad()

            if args.inception:
                class_output, class_aux_output = cls(images)
                loss1 = criterion(class_output, labels)
                loss2 = criterion(class_aux_output, labels)
                loss = loss1 + 0.4*loss2
                loss.backward()

            else:
                class_output = cls(images)
                loss = criterion(class_output, labels)
                loss.backward()

            pred = class_output.max(1, keepdim=True)[1]
            correct = pred.eq(labels.view_as(pred)).sum().item()

            optimizer.step()

            losses.update(loss.data.item(), args.batch_size)
            train_acc.update(correct, args.batch_size)
            train_pbar.update()

            train_pbar.set_postfix({'loss':'{:.4f}'.format(losses.avg),
                                    'acc':'{:.4f}'.format(train_acc.acc),
                                    })

        train_pbar.close()
        if args.test:
            test_acc, test_loss = test(cls, test_loader, args)

            if test_acc.acc > best_acc:
                best_acc = test_acc.acc
                patience = 0
                if len(args.source) > 1:
                    checkpoint_path = join(os.getcwd(), 'models_inception', 'src_combine_tar_{}.pth'.format(args.target[0]))
                else:
                    checkpoint_path = join(os.getcwd(), 'models_inception', 'src_{}.pth'.format(args.source[0]))
                save_checkpoint(checkpoint_path, cls, optimizer)
            else:
                patience += 1

            err_log.write('Loss: {:.4f}/{:.4f}, Accuracy: {:.4f}/{:.4f}\n'.format(losses.avg, test_loss.avg,
                          train_acc.acc, test_acc.acc))
        else:
            if train_acc.acc > best_acc:
                best_acc = train_acc.acc
                if len(args.source) > 1:
                    checkpoint_path = join(os.getcwd(), 'models', 'src_combine_tar_{}.pth'.format(args.target[0]))
                else:
                    checkpoint_path = join(os.getcwd(), 'models', 'src_{}.pth'.format(args.source[0]))
                save_checkpoint(checkpoint_path, cls, optimizer)

            err_log.write('Loss: {:.4f}, Accuracy: {:.4f}\n'.format(losses.avg, train_acc.acc))

        err_log.flush()

        if patience == args.early_stopping:
            break

    err_log.write('Best test_acc: {:.4f}\n'.format(best_acc))
    err_log.close()

if __name__ == '__main__':
    main()
