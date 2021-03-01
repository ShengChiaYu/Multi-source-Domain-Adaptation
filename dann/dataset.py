import os
from os.path import join
import sys
import glob
import math

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


titles = ['usps', 'svhn', 'mnistm']


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

            return image, label


    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len


class DigitsImages(Dataset):
    def __init__(self, root, title, phrase='train', transform=None, pred_only=False):
        """ Intialize the Digits Images dataset """
        self.root = root
        self.title = title
        self.phrase = phrase
        self.transform = transform
        self.pred_only = pred_only
        self.num_class = 10

        # read labels
        if self.pred_only:
            self.image_paths = []
            filenames = glob.glob(os.path.join(self.root, '*.png'))
            for path in sorted(filenames):
                self.image_paths.append(path)

        else:
            self.labels = pd.read_csv(os.path.join(self.root, self.title, '{}.csv'.format(self.phrase)))
            self.image_ids = self.labels['image_name']
            self.labels = self.labels['label']
            self.image_paths = [(os.path.join(root, title, phrase) + '/' + id) for id in self.image_ids]


        # get dataset size
        self.len = len(self.image_paths)


    def __getitem__(self, index):
        """ Get a sample from the dataset """
        # get image
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)

        if self.pred_only:
            return image

        else:
            label = int(self.labels[index])

            return image, torch.tensor(label)

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len


def test(data_dir):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])

    trainset = DigitsImages(root=data_dir, title='usps', phrase='train', transform=transform)
    trainset_loader = DataLoader(trainset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    for batch_idx, (x_imgs, y_target) in enumerate(trainset_loader):
        print (batch_idx, x_imgs.shape, y_target.shape)
        print (y_target)
        break


if __name__ == '__main__':
    data_dir = '../hw3_data/digits/'
    test(data_dir)
