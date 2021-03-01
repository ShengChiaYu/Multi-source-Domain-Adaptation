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
from model import resnet_345, Domain_classifier


def resume_pretrained(resume_path, model):
    state = torch.load(resume_path)
    model.CNN.load_state_dict(state['cnn'])
    model.fc.load_state_dict(state['fc'])
    print ('model loaded from %s' % resume_path)

    return model


def fixed_pretrained(model, fixed_child):
    for name, child in model.named_children():
        if name == fixed_child:
            for param in child.parameters():
                param.requires_grad = False


def test(cls, test_loader, batchSize):
    criterion = nn.CrossEntropyLoss()
    cls.eval()
    losses, test_acc = AverageMeter(), AverageMeter()
    test_pbar = tqdm(total=len(test_loader), ncols=100, leave=True)
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.cuda(), labels.cuda()

            class_output = cls.predict(images)
            loss = criterion(class_output, labels)
            _, pred = torch.max(class_output, 1)
            correct = sum(pred == labels).float() / float(pred.shape[0])

            losses.update(loss.data.item(), batchSize)
            test_acc.update(correct, batchSize)
            test_pbar.update()

            test_pbar.set_postfix({'loss':'{:.4f}'.format(losses.avg),
                                    'acc':'{:.4f}'.format(test_acc.avg),
                                    })
            if (i / len(test_loader)) >= 0.5:
                break

    test_pbar.close()

    return test_acc, losses


def save_checkpoint(checkpoint_path, CNN, cls, optimizer_CNN, optimizer_cls):
    state = {'CNN_state_dict': CNN.state_dict(),
             'cls_state_dict': cls.state_dict(),
             'optimizer_CNN' : optimizer_CNN.state_dict(),
             'optimizer_cls' : optimizer_cls.state_dict(),
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
        drop_last=True,
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
        drop_last=True,
        )

    # models
    src_CNN = resnet_345(args).cuda()
    tar_CNN = resnet_345(args).cuda()
    D_cls = Domain_classifier().cuda()

    # load and fix pretrained models
    resume_path = join(os.getcwd(), args.model_dir, args.model_name)
    resume_pretrained(resume_path, src_CNN)
    resume_pretrained(resume_path, tar_CNN)

    fixed_pretrained(src_CNN, 'CNN')
    fixed_pretrained(src_CNN, 'fc')
    fixed_pretrained(tar_CNN, 'fc')

    # optimizer
    if args.SGD:
        optimizer_tar_CNN = optim.SGD(tar_CNN.parameters(),  lr=args.lr, momentum=0.9)
        optimizer_D_cls = optim.SGD(D_cls.parameters(),  lr=args.lr, momentum=0.9)
    elif args.Adam:
        optimizer_tar_CNN = optim.Adam(tar_CNN.parameters(), lr=args.lr)
        optimizer_D_cls = optim.Adam(D_cls.parameters(), lr=args.lr)

    # domain labels
    src_label = torch.full((args.batch_size,), 0).long().cuda()
    tar_label = torch.full((args.batch_size,), 1).long().cuda()

    min_len = min(len(src_loader), len(tar_loader))
    best_acc = 0
    patience = 0
    err_log_path = join(os.getcwd(), 'models', 'adda_{}_err.txt'.format(args.target[0]))
    err_log = open(err_log_path, 'w')
    for epoch in range(args.start_epoch, args.max_epochs+1):
        criterion = nn.CrossEntropyLoss()
        src_CNN.train()
        tar_CNN.eval()
        D_cls.train()

        print ('\nEpoch = {}'.format(epoch))
        err_log.write('Epoch = {}, '.format(epoch))

        losses, train_acc, D_losses, D_acc = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        train_pbar = tqdm(total=min_len, ncols=100, leave=True)
        for i, (src_data, tar_data) in enumerate(zip(src_loader, tar_loader)):

            src_imgs, _ = src_data
            tar_imgs, tar_labels = tar_data

            src_imgs, tar_imgs, tar_labels = src_imgs.cuda(), tar_imgs.cuda(), tar_labels.cuda()

            # train D_cls
            D_cls.zero_grad()

            src_feature = src_CNN(src_imgs)
            tar_feature = tar_CNN(tar_imgs)

            pred_src_domain = D_cls(src_feature.detach())
            pred_tar_domain = D_cls(tar_feature.detach())
            domain_output = torch.cat((pred_src_domain, pred_tar_domain), 0)
            label = torch.cat((src_label, tar_label), 0)
            domain_output = torch.squeeze(domain_output)

            D_loss = criterion(domain_output, label)
            D_loss.backward()

            optimizer_D_cls.step()

            # domain accuracy
            _, pred = torch.max(domain_output, 1)
            D_correct = sum(pred == label).float() / float(pred.shape[0])

            # train tar_CNN
            tar_CNN.zero_grad()

            tar_feature = tar_CNN(tar_imgs)

            pred_tar_domain = D_cls(tar_feature)
            pred_tar_domain = torch.squeeze(pred_tar_domain)

            loss = criterion(pred_tar_domain, src_label)
            loss.backward()

            optimizer_tar_CNN.step()

            # predict accuracy
            class_output = tar_CNN.predict(tar_imgs)
            _, pred = torch.max(class_output, 1)
            cls_correct = sum(pred == tar_labels).float() / float(pred.shape[0])

            # update losses, accuracy and pbar
            D_losses.update(D_loss.data.item(), args.batch_size*2)
            D_acc.update(D_correct, args.batch_size*2)
            losses.update(loss.data.item(), args.batch_size)
            train_acc.update(cls_correct, args.batch_size)
            train_pbar.update()

            train_pbar.set_postfix({'D_loss':'{:.3f}'.format(D_losses.avg),
                                    'D_acc':'{:.3f}'.format(D_acc.avg),
                                    'loss':'{:.3f}'.format(losses.avg),
                                    'acc':'{:.3f}'.format(train_acc.avg),
                                    })
        train_pbar.close()

        test_acc, test_loss = test(tar_CNN, tar_loader, args.batch_size)
        if test_acc.avg > best_acc:
            best_acc = test_acc.avg
            patience = 0
            checkpoint_path = join(os.getcwd(), 'models', 'adda_{}.pth'.format(args.target[0]))
            save_checkpoint(checkpoint_path, tar_CNN, D_cls, optimizer_tar_CNN, optimizer_D_cls)
        else:
            patience += 1

        err_log.write('Loss: {:.4f}, Accuracy: {:.4f}\n'.format(losses.avg, test_acc.avg))
        err_log.flush()

        if patience >= args.early_stopping:
            print('Early stopping...')
            break

    err_log.write('Best test_acc: {:.4f}\n'.format(best_acc))
    err_log.close()

if __name__ == '__main__':
    main()
