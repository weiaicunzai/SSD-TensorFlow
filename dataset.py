import os
import pickle
import numpy as np


def cifar100_train(data_dir):

    with open(os.path.join(data_dir, 'train'), 'rb') as cifar100_train:
        data = pickle.load(cifar100_train, encoding='bytes')
    
    images = data['data'.encode()]
    labels = data['fine_labels'.encode()]
    images = np.reshape(images, (-1, 3, 32, 32))
    images = np.transpose(images, (0, 2, 3, 1))
    
    return images, labels

def cifar100_test(data_dir):
    with open(os.path.join(data_dir, 'test'), 'rb') as cifar100_test:
        data = pickle.load(cifar100_test, encoding='bytes')

    images = data['data'.encode()]
    labels = data['fine_labels'.encode()] 
    images = np.reshape(images, (-1, 3, 32, 32))
    images = np.transpose(images, (0, 2, 3, 1))

    return images, labels


cifar100_train('/home/admin-bai/Downloads/cifar-100-python')