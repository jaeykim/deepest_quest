import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob
import pandas as pd
import os
import sys
import re
import random
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from skimage import io
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from torch.utils.data import (Dataset, DataLoader)  # Gives easier dataset managment and creates mini batches
from torchvision.datasets import ImageFolder
# from collections import OrderedDict

from model import Net
from train import (train, inference, ensemble_train, ensemble_inference)
from loader import (load_data, load_sep_data, load_model)
import argparse

''' arg parser '''
parser = argparse.ArgumentParser(description='Wildlife DNN model.')
parser.add_argument('--epochs', required=False, default=200, type=int,
                    help='set an epoch (defualt: 200)')
parser.add_argument('--optimizer', required=False, default='sgd',
                    help='select an optimizer (defualt: sgd)')
parser.add_argument('--batch_size', required=False, default=100, type=int,
                    help='batch size (defualt: 100)')
parser.add_argument('--augmentation', required=False, default=True,
                    help='data augmentation (default: True)')
parser.add_argument('--valid_set', required=False, default=10, type=int,
                    help='portion of validation set (default: 10%)')
parser.add_argument('--ensemble', required=False, default=False,
                    help='enable ensemble? (default: False)')
parser.add_argument('--ssl', required=False, default=False,
                    help='semi-supervised learning (default: False)')
parser.add_argument('--ssl_threshold', required=False, default=3, type=int,
                    help='ssl score threshold (default: 3)')
parser.add_argument('--ssl_bypass', required=False, default=False,
                    help='bypass preprocessing of ssl (default: False)')
parser.add_argument('--casename', required=False, default='default',
                    help='case name (default: default)')
parser.add_argument('--debug', required=False, default=False,
                    help='debug mode (default: False)')

def main(argv):
    args = parser.parse_args()
    n_epoch = args.epochs
    optimz = args.optimizer
    batch_size = args.batch_size
    augmentation = args.augmentation
    valid_set = args.valid_set
    ssl = args.ssl
    ssl_threshold = args.ssl_threshold
    ssl_bypass = args.ssl_bypass
    ensemble = args.ensemble
    casename = '{}_{}'.format(args.casename, optimz)
    filename = 'ssl_{}'.format(casename) if ssl else casename
    filename = '{}_ensemble'.format(filename) if ensemble else filename
    debug = args.debug
    n_ensemble = 3
    print('=' * 100)
    print('[ {} ]\noptimizer: {}, semi-supervised learning: {}, ensemble: {}, debug: {}'.format(filename, optimz, ssl, ensemble, debug))

    if not os.path.isfile('./results_{}.txt'.format(filename)):
        with open('./results_{}.txt'.format(filename), 'w') as file:
            if not debug:
                file.write('epoch\ttrain_loss\tvalid_loss\ttest_loss\ttest_accuracy\n')

    # data preprocessing
    PATH = '/data/' # absolute path
    dir = PATH + 'oregon_wildlife/oregon_wildlife'
    files = [f for f in glob(dir + "**/**", recursive=True)] # create a list will allabsolute path of all files
    df_animals = pd.DataFrame({"file_path":files}) # transform in a dataframe
    df_animals['animal'] = df_animals['file_path'].str.extract('oregon_wildlife/oregon_wildlife/(.+)/') # extract the name of the animal
    df_animals['file'] = df_animals['file_path'].str.extract('oregon_wildlife/.+/(.+)') # extrat the file name
    df_animals = df_animals.dropna() # drop nas

    animal_set = set(df_animals['animal'])

    # for index, row in df_animals.iterrows():
    #     print(row['animal'])

    train_list = [PATH + s for s in load_data('./deepest_train.txt')]
    test_list = [PATH + s for s in load_data('./deepest_test.txt')]

    print('train list: {}, test list: {}'.format(len(train_list), len(test_list)))

    # divide the train data into train / valf
    df_animals['train_val_test'] = -1

    df_animals.loc[df_animals['file_path'].isin(train_list), 'train_val_test'] = 0
    df_animals.loc[df_animals['file_path'].isin(test_list), 'train_val_test'] = 2
    # print(df_animals)

    if ensemble:
        en_loaders = []
        for i in range(n_ensemble):
            (loaders, animal_list) = load_sep_data(df_animals=df_animals,
                            train_list=train_list,
                            test_list=test_list,
                            animal_set=animal_set,
                            dir=dir,
                            valid_set=valid_set,
                            augmentation=augmentation,
                            batch_size=batch_size,
                            ssl=False)
            en_loaders.append(loaders)
    else:
        (loaders, animal_list) = load_sep_data(df_animals=df_animals,
                                train_list=train_list,
                                test_list=test_list,
                                animal_set=animal_set,
                                dir=dir,
                                augmentation=augmentation,
                                batch_size=batch_size,
                                ssl=False)
    # print(animal_list)

    # instantiate the CNN
    if ensemble:
        en_models = [Net(), Net(), Net()]
    else:
        model = Net()
        print(model)
        print('Total params: {:,}'.format(sum(p.numel() for p in model.parameters())))
        print('Trainable params: {:,}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        print('-' * 100)

    # set epochs
    start_epoch = 1

    # specify optimizer
    if ensemble:
        en_optimizers = []
        for i in range(n_ensemble):
            if optimz == 'sgd':
                en_optimizers.append(optim.SGD(en_models[i].parameters(), lr=0.001, momentum=0.9))
            elif optimz == 'adam':
                en_optimizers.append(optim.Adam(en_models[i].parameters(), lr=0.0001))
            elif optimz == 'adagrad':
                en_optimizers.append(optim.Adagrad(en_models[i].parameters(), lr=0.0001))
            elif optimz == 'rmsprop':
                en_optimizers.append(optim.RMSprop(en_models[i].parameters(), lr=0.0001))
    else:
        if optimz == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        elif optimz == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=0.0001)
        elif optimz == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=0.0001)
        elif optimz == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=0.0001)

    # specify loss function
    criterion = nn.CrossEntropyLoss()

    # move tensors to GPU if CUDA is available
    use_cuda = torch.cuda.is_available()
    if use_cuda and torch.cuda.device_count() > 1:
        device = torch.device("cuda:0")
        if ensemble:
            for i in range(n_ensemble):
                model = nn.DataParallel(en_models[i])
                model.to(device)
        else:
            model = nn.DataParallel(model)
            model.to(device)

    # load the existing model
    if ensemble:
        for i in range(n_ensemble):
            (loaded, model, optimizer, start_epoch) = load_model('model_{}_{}.pt'.format(filename, i), en_models[i], en_optimizers[i])
            if not loaded and ssl:
                (loaded, model, optimizer, start_epoch) = load_model('model_{}_{}.pt'.format(casename, i), en_models[i], en_optimizers[i])
                # _model = train(n_epoch, start_epoch, loaders, _model, _optimizer, criterion, use_cuda, casename, debug)
            en_models[i] = model
            en_optimizers[i] = optimizer
    else:
        (loaded, model, optimizer, start_epoch) = load_model('model_{}.pt'.format(filename), model, optimizer)
        if not loaded and ssl:
            (loaded, model, optimizer, start_epoch) = load_model('model_{}.pt'.format(casename), model, optimizer)
            model = train(n_epoch, start_epoch, loaders, model, optimizer, criterion, use_cuda, casename, ensemble, debug)
        else:
            (loaded, model, optimizer, start_epoch) = load_model('model_{}.pt'.format(filename), model, optimizer)

    if ensemble:
        if not ssl_bypass:
            en_models = ensemble_train(n_epoch, start_epoch, en_loaders, en_models, en_optimizers, n_ensemble, criterion, use_cuda, filename, debug)
        # semi-supervised learning
        if ssl:
            print('+' * 100)
            print('start semi-supervised learning')
            # ssl_dir = PATH + 'oregon_wildlife/{}'.format(filename)
            # sslExist = os.path.isdir(ssl_dir)

            # labels = None
            # if not sslExist:
            #     os.mkdir(ssl_dir)
            #     for an in animal_set:
            #         os.system('mkdir {}/{}'.format(ssl_dir, an))

            #     labels = ensemble_inference(loaders, en_models, n_ensemble, criterion, ssl_threshold, use_cuda)
            #     print('################ new data: {}'.format(len(np.where(labels > -1)[0])))
            labels = ensemble_inference(loaders, en_models, n_ensemble, criterion, ssl_threshold, use_cuda)

            en_loaders = []
            for i in range(n_ensemble):
                (loaders, animal_list) = load_sep_data(df_animals=df_animals,
                                    train_list=train_list,
                                    test_list=test_list,
                                    animal_set=animal_set,
                                    dir=dir,
                                    augmentation=augmentation,
                                    ssl=True,
                                    # ssl_dir=ssl_dir,
                                    animal_list=animal_list,
                                    labels=labels,
                                    # sslExist=sslExist,
                                    filename=filename)
                en_loaders.append(loaders)
                # sslExist = True

            # train the model
            en_models = ensemble_train(n_epoch, start_epoch, en_loaders, en_models, en_optimizers, n_ensemble, criterion, use_cuda, filename, debug)
    else:
        # semi-supervised learning
        if ssl:
            # ssl_dir = PATH + 'oregon_wildlife/{}'.format(filename)
            # sslExist = os.path.isdir(ssl_dir)

            # if not sslExist:
            #     os.mkdir(ssl_dir)
            #     for an in animal_set:
            #         os.system('mkdir {}/{}'.format(ssl_dir, an))

            # labels = None
            # if not sslExist:
            #     labels = inference(loaders, model, criterion, ssl_threshold, use_cuda)
            #     print('################ new data: {}'.format(len(np.where(labels > -1)[0])))
            labels = inference(loaders, model, criterion, ssl_threshold, use_cuda)

            (loaders, animal_list) = load_sep_data(df_animals=df_animals,
                                    train_list=train_list,
                                    test_list=test_list,
                                    animal_set=animal_set,
                                    dir=dir,
                                    augmentation=augmentation,
                                    ssl=True,
                                    # ssl_dir=ssl_dir,
                                    animal_list=animal_list,
                                    labels=labels,
                                    # sslExist=sslExist,
                                    filename=filename)

            # train the model
            model = train(n_epoch, 1, loaders, model, optimizer, criterion, use_cuda, filename, ensemble, debug)
        else:
            model = train(n_epoch, start_epoch, loaders, model, optimizer, criterion, use_cuda, filename, ensemble, debug)

if __name__ == "__main__":
    main(sys.argv)
