import os

import numpy as np
import torch
from autoaugment import *
from torchvision import datasets, transforms

# from datasets.cifar_c import get_CIFAR10_C, get_CIFAR100_C
# from datasets.autoaugment import CIFAR10Policy


DATA_PATH = '../data/'
# CIFAR_C = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform',
#            'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur',
#            'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate',
#            'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise', 'zoom_blur']

class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class MultiDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform(sample)
        return x1, x2


def get_transform(augment_type):
    image_size = 32

    if augment_type in ['base', 'autoaug_sche']:
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    elif augment_type == 'autoaug':
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
        ])
        train_transform.transforms.append(CutoutDefault(int(image_size / 2)))
    else:
        raise NotImplementedError()

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return train_transform, test_transform


def get_dataset(augment, download=False):
    train_transform, test_transform = get_transform(augment)
    train_transform = MultiDataTransform(train_transform)
    test_transform =  MultiDataTransform(test_transform)

    image_size = (3, 32, 32)
    n_classes = 10
    train_set = datasets.CIFAR10(DATA_PATH, train=True, download=download, transform=train_transform)
    test_set = datasets.CIFAR10(DATA_PATH, train=False, download=download, transform=test_transform)

    return train_set, test_set, image_size, n_classes
    
