# encoding: utf-8
"""
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: train.py.py
@time: 2019/6/20 下午3:59
@desc: train
"""

try:
    import os
    import argparse
    import numpy as np
    import torch.nn as nn
    import torch.nn.functional as F
    import torch
    from dec.models import Encoder_CNN, Decoder_CNN, DEC
    from dec.datasets import dataset_list, get_dataloader
    from dec.definitions import RUNS_DIR, DATASETS_DIR
    from itertools import chain as ichain
    from sklearn.cluster import KMeans
    import sklearn.metrics as metrics
except ImportError as e:
    print(e)
    raise ImportError


def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    parser.add_argument("-r", "--run_name", dest="run_name", default="vaegan", help="Name of training run")
    parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=200, type=int, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=256, type=int, help="Batch size")
    parser.add_argument("-s", "--dataset_name", dest="dataset_name", default='mnist', choices=dataset_list,
                        help="Dataset name")
    parser.add_argument("-p", "--pretrain", dest="pretrain", default=True, help="pretrain ae")
    args = parser.parse_args()

    run_name = args.run_name
    dataset_name = args.dataset_name

    # make directory
    run_dir = os.path.join(RUNS_DIR, dataset_name, run_name)
    data_dir = os.path.join(DATASETS_DIR, dataset_name)
    imgs_dir = os.path.join(run_dir, 'images')
    models_dir = os.path.join(run_dir, 'models')
    log_path = os.path.join(run_dir, 'logs')

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    #------train var-------
    sgd_lr = 0.01
    momentum = 0.9
    epochs = args.n_epochs
    batch_size = args.batch_size
    pretrain_epochs = 300
    pretrain = args.pretrain
    n_cluster = 10
    lr_adam = 1e-4 # 0.0001, 用于预训练autoencoder
    b1 = 0.5
    b2 = 0.9 #99
    decay = 2.5*1e-5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #------test var--------
    test_batch_size = 1500

    # net
    encoder = Encoder_CNN(n_cluster=n_cluster, batch_size=batch_size)
    decoder = Decoder_CNN(n_cluster=n_cluster, batch_size=batch_size, img_feature=(1, 28, 28))
    dec = DEC(n_cluster=n_cluster, batch_size=batch_size)

    # 组合参数
    autoencoder_params = ichain(encoder.parameters(),
                                decoder.parameters())

    # optimization
    auto_op = torch.optim.Adam(autoencoder_params, lr=lr_adam, betas=(b1, b2), weight_decay=decay)
    # dec_op = torch.optim.SGD(dec.parameters(), lr=sgd_lr, momentum=momentum)

    # dataloader
    dataloader = get_dataloader(dataset_path=data_dir, dataset_name=dataset_name, batch_size=batch_size, train=True)

    # loss
    auto_loss = nn.MSELoss()

    # to cuda
    encoder.to(device)
    decoder.to(device)
    dec.to(device)
    auto_loss.to(device)

    # ----pretrain----
    if pretrain:
        print('...Pretraining...')
        for epoch in range(pretrain_epochs):
            for i, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                encoder.train()
                decoder.train()
                encoder.zero_grad()
                decoder.zero_grad()

                z = encoder(data)
                output = decoder(z)

                loss = auto_loss(output, output)
                loss.backward()

                auto_op.step()

        # save model params
        torch.save(encoder.state_dict(), os.path.join(models_dir, 'encoder.pkl'))
        torch.save(decoder.state_dict(), os.path.join(models_dir, 'decoder.pkl'))

        # caculate acc of autoencoder
        # encoder.eval()
        # features = encoder.predict(data)
        # km = KMeans(n_clusters=n_cluster, n_init=20, n_jobs=4)
        # y_pred = km.fit_predict(features)
        # print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
        #       % (metrics.acc(target, y_pred), metrics.nmi(target, y_pred)))
    else:
        encoder.load_state_dict(torch.load(os.path.join(models_dir, 'encoder.pkl')))
        decoder.load_state_dict((torch.load(os.path.join(models_dir, 'decoder.pkl'))))


if __name__ == '__main__':
    main()