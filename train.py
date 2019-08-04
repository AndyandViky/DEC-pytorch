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
    from dec.utils import save_images, kl_divergence
    from itertools import chain as ichain
    from sklearn.cluster import KMeans
    import dec.metrics as metrics
    import time
except ImportError as e:
    print(e)
    raise ImportError


def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    parser.add_argument("-r", "--run_name", dest="run_name", default="dec", help="Name of training run")
    parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=50, type=int, help="Number of epochs")
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
    pretrain_epochs = 10
    pretrain = args.pretrain
    n_cluster = 10
    lr_adam = 1e-4 # 0.0001, 用于预训练autoencoder
    b1 = 0.5
    b2 = 0.9 #99
    decay = 2.5*1e-5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    alpha = 1.0

    #------test var--------
    test_batch_size = 1500

    # net
    encoder = Encoder_CNN(n_cluster=n_cluster, batch_size=batch_size)
    decoder = Decoder_CNN(n_cluster=n_cluster, batch_size=batch_size, img_feature=(1, 28, 28))

    # 组合参数
    autoencoder_params = ichain(encoder.parameters(),
                                decoder.parameters())

    # optimization
    auto_op = torch.optim.Adam(autoencoder_params, lr=lr_adam, betas=(b1, b2), weight_decay=decay)

    # dataloader
    dataloader = get_dataloader(dataset_path=data_dir, dataset_name=dataset_name, batch_size=batch_size, train=True)

    # loss
    auto_loss = nn.MSELoss()

    # to cuda
    encoder.to(device)
    decoder.to(device)
    auto_loss.to(device)

    # ----pretrain----
    if pretrain:
        print('...Pretraining...')
        logger = open(os.path.join(log_path, "log.txt"), 'a')
        logger.write("pretraining...\n")
        logger.close()
        t0 = time()
        for epoch in range(pretrain_epochs):
            for i, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                encoder.train()
                decoder.train()
                encoder.zero_grad()
                decoder.zero_grad()

                z = encoder(data)
                output = decoder(z)

                loss = auto_loss(data, output)
                loss.backward()

                auto_op.step()

            # caculate acc of autoencoder
            data, target = next(iter(dataloader))
            data = data.to(device)
            encoder.eval()
            decoder.eval()
            features = encoder(data)
            save_images(decoder(features), "%s/%s_%d.png" % (imgs_dir, 'pretrain_test', epoch))
            data, target = features.data.cpu(), target.numpy()
            km = KMeans(n_clusters=n_cluster, n_init=20)
            y_pred = km.fit_predict(features.data.cpu())
            acc = metrics.acc(target, y_pred)
            nmi = metrics.nmi(target, y_pred)
            print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
                  % (acc, nmi))

            # log pretrain info
            logger = open(os.path.join(log_path, "log.txt"), 'a')
            logger.write(
                "iter={}, acc={:.4f}, nmi={:.4f}\n".format(
                    epoch, acc, nmi
                )
            )
            logger.close()

        t1 = time()
        print("pretrain time: %d" % t1-t0)
        # save model params
        torch.save(encoder.state_dict(), os.path.join(models_dir, 'encoder.pkl'))
        torch.save(decoder.state_dict(), os.path.join(models_dir, 'decoder.pkl'))

    else:
        encoder.load_state_dict(torch.load(os.path.join(models_dir, 'encoder.pkl')))
        decoder.load_state_dict((torch.load(os.path.join(models_dir, 'decoder.pkl'))))


    #============================DEC===============================

    dec = DEC(encoder=encoder, n_cluster=n_cluster, batch_size=batch_size, alpha=alpha)
    dec_op = torch.optim.SGD(dec.parameters(), lr=sgd_lr, momentum=momentum)
    dec.to(device)

    # init mu
    data, target = next(iter(dataloader))
    dec.get_assign_cluster_centers_op(encoder(data.to(device)))

    logger = open(os.path.join(log_path, "log.txt"), 'a')
    logger.write("============================DEC===============================\n")
    logger.close()

    for epoch in epochs:

        for i, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            dec.train()
            dec.zero_grad()
            dec_op.zero_grad()

            q, p = dec(data)
            loss = kl_divergence(p, q)

            loss.backward()
            dec_op.step()

        # test
        _data, _target = next(iter(dataloader))
        q, p = dec(_data)
        pred = torch.argmax(q, dim=1)
        print("[DEC] epoch: {}\tloss: {}\tacc: {}".format(epoch, loss, metrics.acc(target, pred)))
        logger = open(os.path.join(log_path, "log.txt"), 'a')
        logger.write("[DEC] epoch: {}\tloss: {}\tacc: {}\n"
                     .format(epochs, loss, metrics.acc(target, pred)))
        logger.close()

    # save dec model
    torch.save(dec.state_dict(), os.path.join(models_dir, 'dec.pkl'))


if __name__ == '__main__':
    main()