# encoding: utf-8
"""
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: models.py
@time: 2019/6/20 下午4:00
@desc: models
"""

try:
    import torch
    import numpy as np
    import torch.nn as nn
    import torchvision
    from torch.autograd import Variable
    from vaegan.utils import init_weights
except ImportError as e:
    print(e)
    raise ImportError


class Reshape(nn.Module):
    def __init__(self, reshape=[]):
        super(Reshape, self).__init__()

        self.reshape = reshape

    def forward(self, x):
        x = x.view(x.size(0), *self.reshape)
        return x

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'shape={}'.format(
            self.reshape
        )


class Encoder_CNN(nn.Module):
    """
    encoder module
    input: reak-image
    output: vector sample from z
    """
    def __init__(self, batch_size=60, latent_dim=50, verbose=False):
        super(Encoder_CNN, self).__init__()

        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.cshape = (128, 7, 7)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.verbose = verbose

        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            Reshape(self.lshape),

            nn.Linear(self.iels, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),

            nn.Linear(1024, self.latent_dim)
        )

        init_weights(self)

        if self.verbose:
            print(self.model)

    def forward(self, x):
        # input size [batch_size, 1, 28, 28]
        # output size [batch_size, latent_dim]
        z = self.model(x)
        return z


class Decoder_CNN(nn.Module):
    def __init__(self, latent_dim=50, batch_size=60, img_feature=(1, 28, 28), verbose=False):
        super(Decoder_CNN, self).__init__()

        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.img_feature = img_feature
        self.cshape = (128, 7, 7)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.verbose = verbose

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),

            nn.Linear(1024, self.iels),
            nn.BatchNorm1d(self.iels),
            nn.ReLU(True),

            Reshape(self.cshape),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

        init_weights(self)

        if self.verbose:
            print(self.model)

    def forward(self, x):
        # input size [batch_size, latent_dim]
        # output size [batch_size, 1, 28, 28]
        gen_imgs = self.model(x)
        gen_imgs = gen_imgs.view(gen_imgs.size(0), *self.img_feature)
        return gen_imgs


class Discrinimator_CNN(nn.Module):
    def __init__(self, verbose=False):
        super(Discrinimator_CNN, self).__init__()

        self.cshape = (128, 7, 7)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.verbose = verbose

        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),

            Reshape(self.lshape),

            nn.Linear(self.iels, 1024),
            nn.ReLU(True),

            nn.Linear(1024, 1),
        )

        init_weights(self)

        if self.verbose:
            print(self.model)

    def forward(self, x):
        valid = self.model(x)
        return valid


# def img_show(img):
#     img = img / 2 + 0.5
#     nimg = img.numpy()
#     plt.imshow(np.transpose(nimg, (1, 2, 0)))
#     plt.show()
#
#
# encoder = Encoder_CNN()
# decoder = Decoder_CNN()
# from vaegan.datasets import get_dataloader
# import matplotlib.pyplot as plt
# dataloader = get_dataloader()
# real_imgs, target = next(iter(dataloader))
# z = encoder(real_imgs)
#
# fake_imgs = decoder(z)
#
# img_show(torchvision.utils.make_grid(fake_imgs.data))
