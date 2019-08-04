# encoding: utf-8
"""
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: utils.py
@time: 2019/6/20 下午4:00
@desc: utils function
"""

try:
    import matplotlib.pyplot as plt
    import torchvision
    import numpy as np
    import torch
    import torch.nn as nn
except ImportError as e:
    print(e)
    raise ImportError


def img_show(img):
    img = img / 2 + 0.5
    nimg = img.numpy()
    plt.imshow(np.transpose(nimg, (1, 2, 0)))
    plt.show()


def init_weights(Net):
    for m in Net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


def save_images(tensor, filename):
    torchvision.utils.save_image(
        tensor.data[:25],
        filename,
        nrow=5,
        normalize=True
    )


def kl_divergence(target, pred):
    return torch.mean(torch.sum(target * torch.log(target/pred), dim=1))

