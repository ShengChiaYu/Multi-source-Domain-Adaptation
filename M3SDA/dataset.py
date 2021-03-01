import os
import sys
import glob
import argparse
from os.path import join

import random
import numpy as np
import pandas as pd

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from utils import parse_args


class dataset_public(Dataset):
    def __init__(self, root, transform=None, train=True, domain=None, real_test=False):
        """ Intialize the Face dataset """
        self.root = root
        self.transform = transform
        self.data = []
        self.real_test = True if domain == 'real' and train == False else False

        # read filenames
        if self.real_test:
            self.data = sorted(glob.glob(join(root, 'test/**')))

            self.len = len(self.data)
            print('Number of samples:',self.len)

        else:
            if train:
                print('Training data:', domain)
                domain_root = join(self.root, domain, '{}_train.csv'.format(domain))
                self.data = pd.read_csv(domain_root)

            else:
                print('Testing data:', domain)
                domain_root = join(self.root, domain, '{}_test.csv'.format(domain))
                self.data = pd.read_csv(domain_root)

            # number of samples
            self.len = len(self.data.index)
            print('Number of samples:',self.len)


    def __getitem__(self, index):
        """ Get a sample from the dataset """
        if self.real_test:
            image_fn = self.data[index]
            image = Image.open(image_fn)

            if self.transform is not None:
                image = self.transform(image)

            return image, image_fn.split('/')[-1]

        else:
            image_fn = self.data.loc[index]['image_name']
            image = Image.open(join(self.root, image_fn))
            label = self.data.loc[index]['label']

            if self.transform is not None:
                image = self.transform(image)

            return image, image_fn.split('/')[-1]


    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len


def test():
    # parse args
    args = parse_args()

    dataroot = join(os.getcwd(), args.data_dir)
    transform=transforms.Compose([
              # transforms.RandomCrop(args.img_size, padding=None, pad_if_needed=True, fill=0, padding_mode='edge'),
              transforms.Resize((args.img_size,args.img_size)),
              transforms.ToTensor(),
              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = dataset_public(root=dataroot,
                             transform=transform,
                             train=False,
                             # domain=args.source,
                             # domain=args.target,
                             domain=['infograph', 'sketch', 'quickdraw']
                             # real_test=True,
    )

    # test one sample
    # image, label = dataset.__getitem__(11)
    # image, image_fn = dataset.__getitem__(11)
    # image.show()
    # print(image.size)
    # print(label)
    # print(image_fn)

    # test a batch
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16,
                                             shuffle=True, num_workers=6)
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    # images = dataiter.next()

    print('Image tensor in each batch:', images.shape, images.dtype)
    print('Label tensor in each batch:', labels.shape, labels.dtype)
    # print(images)
    print(labels)


if __name__ == '__main__':
    test()
