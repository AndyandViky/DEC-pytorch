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
    from dec.utils import init_weights
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
    def __init__(self, batch_size=256, n_cluster=10, verbose=False):
        super(Encoder_CNN, self).__init__()

        self.batch_size = batch_size
        self.n_cluster = n_cluster
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

            nn.Linear(1024, self.n_cluster)
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
    def __init__(self, n_cluster=10, batch_size=256, img_feature=(1, 28, 28), verbose=False):
        super(Decoder_CNN, self).__init__()

        self.n_cluster = n_cluster
        self.batch_size = batch_size
        self.img_feature = img_feature
        self.cshape = (128, 7, 7)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.verbose = verbose

        self.model = nn.Sequential(
            nn.Linear(self.n_cluster, 1024),
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


class Cluster_Layer(nn.Module):
    def __init__(self, n_cluster=10, weights=None, alpha=1.0, **kwargs):
        super(Cluster_Layer, self).__init__(**kwargs)

        self.n_cluster = n_cluster
        self.weights = weights
        self.alpha = alpha
        self.clusters = None

        if self.weights is not None:
            del self.weights
            pass

    def forward(self, x):
        q = 1.0 / (1.0 + (torch.sum(torch.square(
            np.expand_dims(x.data.cup().numpy(), axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = torch.transpose(torch.transpose(q) / torch.sum(q, axis=1))
        return q


class DEC(nn.Module):
    def __init__(self, encoder=None, n_cluster=10, batch_size=256):
        super(DEC, self).__init__()

        assert isinstance(encoder, nn.Module)
        self.encoder = encoder
        self.n_cluster = n_cluster
        self.batch_size = batch_size

    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def forward(self, x):
        features = self.encoder(x)
        cluster = Cluster_Layer()
        q = cluster(features)
        p = self.target_distribution(q)
        return p, q


# encoder = Encoder_CNN()
# decoder = Decoder_CNN()
# from dec.datasets import get_dataloader
# from dec.utils import img_show
# dataloader = get_dataloader()
# real_imgs, target = next(iter(dataloader))
# z = encoder(real_imgs)
#
# fake_imgs = decoder(z)
#
# img_show(torchvision.utils.make_grid(fake_imgs.data))

