import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob
import pandas as pd
import os
import re
import random
import torch
from torch.utils.data import (Dataset, DataLoader)  # Gives easier dataset managment and creates mini batches
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from torchvision.datasets import ImageFolder

# def load_files(filename):
#     data = []
#     f = open(filename, 'r')
#     lines = f.readlines()
#     for line in lines:
#         data.append(line.rstrip('\n'))
#     f.close()
#     return data

# dir = './data/oregon_wildlife/oregon_wildlife'
# files = [f for f in glob(dir + "**/**", recursive=True)] # create a list will allabsolute path of all files

# animal_list = ['bald_eagle', 'black_bear', 'bobcat', 'canada_lynx', 'columbian_black-tailed_deer', 'cougar', 'coyote', 'deer', 'elk', 'gray_fox', 'gray_wolf', 'mountain_beaver', 'nutria', 'raccoon', 'raven', 'red_fox', 'ringtail', 'sea_lions', 'seals', 'virginia_opossum']

def load_data(filepath):
    data = []
    f = open(filepath, 'r')
    lines = f.readlines()
    for line in lines:
        data.append(line.rstrip('\n'))
    f.close()
    return data

    # train_files = load_files(filepath)
    # df_animals = pd.DataFrame({"file_path":train_files}) # transform in a dataframe
    # df_animals['animal'] = df_animals['file_path'].str.extract('/oregon_wildlife/(.+)/') # extract the name of the animal
    # df_animals['file'] = df_animals['file_path'].str.extract('oregon_wildlife/.+/(.+)') # extrat the file name
    # df_animals = df_animals.dropna() # drop nas
    # return df_animals

    # return train_files

def load_sep_data(df_animals, train_list, test_list, animal_set, dir, valid_set=10, augmentation=False, batch_size=100, num_workers=10, ssl=False, animal_list=None, labels=None, filename='default'):

    # move accurate test sets to the train sets
    # print('ssl: {}, animal_list: {}, sslExist: {}'.format(ssl, animal_list, sslExist))
    print('ssl: {}, animal_list: {}'.format(ssl, animal_list))
    if ssl:
        # if sslExist:
        #     print('semi-supervised learning load segmented data')
        #     files = [f for f in glob(ssl_dir + "**/**", recursive=True)] # create a list will allabsolute path of all files
        #     train_df_animals = pd.DataFrame({"file_path":files}) # transform in a dataframe
        #     train_df_animals['animal'] = train_df_animals['file_path'].str.extract('oregon_wildlife/{}/(.+)/'.format(filename)) # extract the name of the animal
        #     train_df_animals['file'] = train_df_animals['file_path'].str.extract('{}/.+/(.+)'.format(filename)) # extrat the file name
        #     train_df_animals = train_df_animals.dropna() # drop nas
        #     train_df_animals['train_val_test'] = 0
        #     print('################ new data: {}'.format(len(train_df_animals.index)))
            
        print('semi-supervised learning data segmentation')
        # print(animal_list)
        # train_df_animals = df_animals.copy()
        df_animals['ssl'] = -1
        df_animals['ssl_animal'] = ''
        n = 0
        for i, row in df_animals.iterrows():
            if row['train_val_test'] > -1 and row['train_val_test'] < 2:
                # os.system('ln -s {} {}/{}'.format(row['file_path'], ssl_dir, row['animal']))
                df_animals.loc[i, 'ssl'] = 0
            elif row['train_val_test'] == 2:
                if labels[n] > -1:
                    # os.system('ln -s {} {}/{}'.format(row['file_path'], ssl_dir, animal_list[labels[n]]))
                    # train_df_animals.at[i, 'train_val_test'] = 0
                    # train_df_animals.at[i, 'file_path'] = '{}/{}/{}'.format(ssl_dir, row['animal'], row['file'])
                    df_animals.loc[i, 'ssl'] = 0
                    df_animals.loc[i, 'ssl_animal'] = animal_list[labels[n]]
                n = n + 1

    # separate train / valid sets
    train_val_test_list = [0,1]
    train_val_weights = [100 - valid_set, valid_set]
    
    if ssl:
        for an in animal_set:
            n = sum((df_animals['ssl'] == 0) & (df_animals['animal'] == an)) # count the number of images of each animal
            train_val_test = random.choices(train_val_test_list, weights= train_val_weights,  k=n)
            df_animals.loc[(df_animals['ssl'] == 0) & (df_animals['animal'] == an), 'ssl'] = train_val_test
        # print('updated df_animals for ssl:')
        # print(df_animals)
   
    for an in animal_set:
        n = sum((df_animals['train_val_test'] == 0) & (df_animals['animal'] == an)) # count the number of images of each animal
        train_val_test = random.choices(train_val_test_list, weights= train_val_weights,  k=n)
        df_animals.loc[(df_animals['train_val_test'] == 0) & (df_animals['animal'] == an), 'train_val_test'] = train_val_test
    print('df_animals:')
    print(df_animals)

    # transform the train/valid data with augmentation
    transform = {
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    if augmentation:
        transform['train'] = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform['train'] = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def check_train(path):
        return (df_animals[df_animals['file_path'] == path].train_val_test == 0).bool()

    def check_valid(path):
        return (df_animals[df_animals['file_path'] == path].train_val_test == 1).bool()

    def check_test(path):
        return (df_animals[df_animals['file_path'] == path].train_val_test == 2).bool()
    
    def ssl_check_train(path):
        return (df_animals[df_animals['file_path'] == path].ssl == 0).bool()

    def ssl_check_valid(path):
        return (df_animals[df_animals['file_path'] == path].ssl == 1).bool()

    if ssl:
        image_datasets = {
            'train' : ImageFolder(root=dir, transform=transform['train'], is_valid_file=ssl_check_train),
            'valid' : ImageFolder(root=dir, transform=transform['valid'], is_valid_file=ssl_check_valid),
            'test' : ImageFolder(root=dir, transform=transform['test'], is_valid_file=check_test),
        }
    else:
        image_datasets = {
            'train' : ImageFolder(root=dir, transform=transform['train'], is_valid_file=check_train),
            'valid' : ImageFolder(root=dir, transform=transform['valid'], is_valid_file=check_valid),
            'test' : ImageFolder(root=dir, transform=transform['test'], is_valid_file=check_test),
        }

    loaders = {
        'train' : DataLoader(image_datasets['train'], shuffle=True, batch_size=batch_size, num_workers=num_workers),
        'valid' : DataLoader(image_datasets['valid'], shuffle=True, batch_size=batch_size, num_workers=num_workers),
        'test' : DataLoader(image_datasets['test'], shuffle=False, batch_size=batch_size, num_workers=num_workers)
    }
    
    animal_list = image_datasets['train'].classes

    print('train: {}'.format(len(loaders['train'].dataset)))
    print('valid: {}'.format(len(loaders['valid'].dataset)))
    print('test: {}'.format(len(loaders['test'].dataset)))

    return loaders, animal_list

def load_model(filename, model, optimizer):
    loaded = False
    model_file = 'model_{}.pt'.format(filename)
    start_epoch = 1
    if os.path.isfile(model_file):
        loaded = True
        os.system('cp {} backup_{}'.format(model_file, model_file))
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load('model_{}.pt'.format(filename), map_location="cuda:0")
        start_epoch = checkpoint['epoch']
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(filename, start_epoch))
    return (loaded, model, optimizer, start_epoch)