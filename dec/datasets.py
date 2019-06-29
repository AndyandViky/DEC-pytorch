# encoding: utf-8
"""
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: datasets.py
@time: 2019/6/20 下午4:00
@desc: datasets
"""

try:
    import torch
    import torchvision.datasets as dset
    import torchvision.transforms as transforms

except ImportError as e:
    raise ImportError


DATASET_FN_DICT = {'mnist': dset.MNIST}


dataset_list = DATASET_FN_DICT.keys()


def get_datasets(dataset_name='mnist'):
    if dataset_name in DATASET_FN_DICT:
        return DATASET_FN_DICT[dataset_name]
    else:
        raise ValueError('dataset_name is not validata')


def get_dataloader(dataset_path='../datasets/mnist', dataset_name='mnist', batch_size=60, train=True):
    dataset = get_datasets(dataset_name)

    dataloader = torch.utils.data.DataLoader(
        dataset(dataset_path, download=True, transform=transforms.Compose([
            transforms.ToTensor()
        ]), train=train),
        batch_size=batch_size,
        shuffle=True
    )

    return dataloader
