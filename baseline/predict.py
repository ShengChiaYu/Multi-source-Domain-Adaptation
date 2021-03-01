import torch.optim as optim
import torch.utils.data
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms

import numpy as np
import random
import os
import sys
from os.path import join, splitext

from tqdm import tqdm, tnrange
from dataset import dataset_public
from utils import AverageMeter, parse_args
from model import resnet_345, inception_345

def load_data(args, train=False):
    torch.manual_seed(args.manual_seed)

    dataroot = join(os.getcwd(), args.data_dir)
    transform=transforms.Compose([
              # transforms.RandomCrop(args.img_size, padding=None, pad_if_needed=True, fill=0, padding_mode='edge'),
              transforms.Resize((args.img_size,args.img_size)),
              transforms.ToTensor(),
              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_dataset = dataset_public(root=dataroot,
                             transform=transform,
                             train=train,
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

    return test_loader


def fixed_pretrained(cls):
    # fix cnn weight
    for name, child in cls.named_children():
        if name == 'CNN':
            for param in child.parameters():
                param.requires_grad = False


def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)


def test_real(cls, test_loader, args, filename):
    cls.eval()
    test_pbar = tqdm(total=len(test_loader), ncols=100, leave=True)
    load_path = os.path.join(os.getcwd(), 'prediction', filename)
    f = open(load_path, 'w')
    f.write('image_name,label\n')
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for i, (images, fn) in enumerate(test_loader):
            images = images.cuda()
            if args.inception:
                class_output, _ = cls(images)
            else:
                class_output = cls(images)

            pred = class_output.max(1, keepdim=True)[1] # get the index of the max log-probability

            for i in range(pred.shape[0]):
                f.write('test/{},{}\n'.format(fn[i], pred[i].data.item()))

            test_pbar.update()
            f.flush()

    test_pbar.close()


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


def main():
    args = parse_args()

    # load real test data
    test_loader = load_data(args, train=False)

    # model
    checkpoint_path = join(os.getcwd(), args.model_dir, args.model_name)
    if args.inception:
        model = inception_345().cuda()
    else:
        model = resnet_345(args).cuda()

    if args.fixed_pretrained:
        fixed_pretrained(model)

    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
    print('Number of parameters to update: {}'.format(len(params_to_update)))

    optimizer = optim.SGD(params_to_update,  lr=args.lr, momentum=0.9)
    load_checkpoint(checkpoint_path, model, optimizer)

    if args.real_test:
        print('model:{}, target:{}'.format(args.model_name, args.target))
        test_real(model, test_loader, args, '{}_result.csv'.format(splitext(args.model_name)[0]))
    else:
        print('model:{}, target:{}'.format(args.model_name, args.target))
        test(model, test_loader, args)

if __name__ == '__main__':
    main()
