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
    from vaegan.models import Encoder_CNN, Decoder_CNN, Discrinimator_CNN
    from vaegan.datasets import dataset_list, get_dataloader
    from vaegan.definitions import RUNS_DIR, DATASETS_DIR
except ImportError as e:
    print(e)
    raise ImportError


def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    parser.add_argument("-r", "--run_name", dest="run_name", default="vaegan", help="Name of training run")
    parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=2000, type=int, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=60, type=int, help="Batch size")
    parser.add_argument("-s", "--dataset_name", dest="dataset_name", default='mnist', choices=dataset_list,
                        help="Dataset name")
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
    #------test var--------


if __name__ == '__main__':
    main()