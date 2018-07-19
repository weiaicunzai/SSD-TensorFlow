import tensorflow as tf
from vgg16_bn import vgg16_bn 
from dataset import *
from tensorflow.python import keras

import matplotlib.pyplot as plt

BATCH_SZIE = 32 
CIFAR100 = "/home/admin-bai/Downloads/cifar-100-python"

datagen = keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    featurewise_center=True,
    rotation_range=15,
    width_shift_range=4,
    featurewise_std_normalization=True
)
images_train, labels_train = cifar100_train('/home/admin-bai/Downloads/cifar-100-python')
labels_train = keras.utils.to_categorical(labels_train, 100)
datagen.fit(images_train)

#optimizer
sgd = keras.optimizers.SGD(lr=0.1, momentum=0.9)
#adam = keras.optimizers.Adam(lr=0.1, decay=1e-6)


#learning_rate decay:
def lr_scheduler(epoch):
    if epoch < 80:
        return 0.1
    elif epoch < 120:
        return 0.01
    else:
        return 0.001


net = vgg16_bn()
net.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
net.fit_generator(
    datagen.flow(images_train, labels_train, batch_size=BATCH_SZIE),
    steps_per_epoch=len(images_train) / 32,
    epochs=10,
    callbacks=[lr_scheduler]
)

net.save_weights('checkpoint/vgg16_bn.h5')

