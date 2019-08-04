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
    from sklearn.cluster import KMeans
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
    def __init__(self, n_cluster=10, weights=None, alpha=1.0, clusters=None, **kwargs):
        super(Cluster_Layer, self).__init__(**kwargs)

        self.n_cluster = n_cluster
        self.weights = weights
        self.alpha = alpha
        self.clusters = clusters

        if self.weights is not None:
            del self.weights
            pass

    def forward(self, x):
        q = 1.0 / (1.0 + (torch.sum(torch.pow(
            torch.unsqueeze(x, 1) - self.clusters, exponent=2), dim=2) / self.alpha))
        q = torch.pow(q, exponent=(self.alpha + 1.0) / 2.0)
        q = torch.transpose(torch.transpose(q, 0, 1) / torch.sum(q, dim=1), 0, 1)
        return q


class DEC(nn.Module):
    def __init__(self, encoder=None, n_cluster=10, batch_size=256, encoder_dim=10, alpha=1.0):
        super(DEC, self).__init__()

        assert isinstance(encoder, nn.Module)
        self.encoder = encoder
        self.n_cluster = n_cluster
        self.batch_size = batch_size
        self.mu = torch.zeros(self.n_cluster, encoder_dim)
        self.kmeans = KMeans(n_clusters=self.n_cluster, n_init=20)
        self.alpha = alpha
        self.cluster_layer = None

    def get_assign_cluster_centers_op(self, features):
        # init mu
        print("Kmeans train start.")
        result = self.kmeans.fit(features.data)
        print("Kmeans train end.")
        self.mu = torch.from_numpy(result.cluster_centers_).repeat(1, 1)
        self.mu = torch.as_tensor(self.mu, dtype=torch.float32)
        self.cluster_layer = Cluster_Layer(n_cluster=self.n_cluster, alpha=self.alpha, clusters=self.mu.cuda())
        return self.mu

    def target_distribution(self, q):
        p = q ** 2 / q.sum(0)
        p = p / p.sum(dim=1, keepdim=True)
        return p

    def forward(self, x):
        features = self.encoder(x)
        q = self.cluster_layer(features)
        p = self.target_distribution(q)
        return q, p

# encoder = Encoder_CNN()
# decoder = Decoder_CNN()
# dec = DEC(encoder=encoder)
# from dec.datasets import get_dataloader
# from dec.utils import img_show
# dataloader = get_dataloader()
# real_imgs, target = next(iter(dataloader))
# z = encoder(real_imgs)
#
# mu = dec.get_assign_cluster_centers_op(z)
#
# p, q = dec(real_imgs)

# img_show(torchvision.utils.make_grid(fake_imgs.data))

