"""
read and preprocess cifar100 data
"""

import tensorflow as tf
import functools
import pickle
import numpy as np
import os


def data_augment(image_dataset, label_dataset):
    """Data argument for dataset, including: random
    ration, random crop, random flip, mean substraction
    and std diviation, transpose of matrix from D/W/H
    to W/H/D
    """

    #reshape to 3, 32, 32
    reshape = functools.partial(tf.reshape, shape=[3, 32, 32])
    image_dataset = image_dataset.map(reshape)

    #transpose to W/H/D format
    transpose = functools.partial(tf.transpose, perm=(1, 2, 0))
    image_dataset = image_dataset.map(transpose)

    #random crop
    pad = functools.partial(tf.pad, paddings=tf.constant([[0, 0], [4, 4], [4, 4]]))
    image_dataset = image_dataset.map(pad)
    crop = functools.partial(tf.random_crop, size=[32, 32, 3])
    image_dataset = image_dataset.map(crop)

    #random flip
    image_dataset = image_dataset.map(tf.image.random_flip_left_right)

    #random rotation
    rotation = functools.partial(tf.contrib.image.rotate, angles=10)
    image_dataset = image_dataset.map(rotation)

    #standard
    image_dataset = image_dataset.map(tf.image.per_image_standardization)
    return image_dataset, label_dataset


def cifar100_train(data_dir, batch_size):
    """Read and return cifar100 training dataset

    Args:
        data_dir: cifar100 dataset path to cifar100
        batch_size: batch_size for cifar100 training
            dataset
    
    Returns: a tensorflow Dataset object contains images
        and labels
    """

    data_dir = os.path.join(data_dir, 'train')
    with open(data_dir, 'rb') as cifar100_train:
        cifar100 = pickle.load(cifar100_train, encoding='bytes')

    images = tf.convert_to_tensor(cifar100['data'.encode()], dtype=tf.float32)
    labels = tf.convert_to_tensor(cifar100['fine_labels'.encode()], dtype=tf.int64)

    image_dataset = tf.data.Dataset.from_tensor_slices(images)
    labels_dataset = tf.data.Dataset.from_tensor_slices(labels)

    image_dataset, labels_dataset = data_augment(image_dataset, labels_dataset)

    dataset = tf.data.Dataset.zip((image_dataset, labels_dataset))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(32)

    return dataset




