from simclr import SimCLR
from model import alexnet
from transform import simclr_transforms

import torch
import torchvision

# import matplotlib.pyplot as plt

# BATCH_SZIE = 32 
# CIFAR100 = "/home/admin-bai/Downloads/cifar-100-python"

#datagen_train = keras.preprocessing.image.ImageDataGenerator(
#    horizontal_flip=True,
#    featurewise_center=True,
#    rotation_range=15,
#    width_shift_range=4,
#    featurewise_std_normalization=True,
#)
#
#images_train, labels_train = cifar100_train('/home/admin-bai/Downloads/cifar-100-python')
#labels_train = keras.utils.to_categorical(labels_train, 100)
#datagen_train.fit(images_train)

#optimizer
#sgd = keras.optimizers.SGD(lr=0.1, momentum=0.9)
#adam = keras.optimizers.Adam(lr=0.1, decay=1e-6)

#learning_rate decay:
#def lr_scheduler(epoch):
#    if epoch < 80:
#        return 0.1
#    elif epoch < 120:
#        return 0.01
#    else:
#        return 0.001

#tb = keras.callbacks.TensorBoard(log_dir='log')
#lr_sch = keras.callbacks.LearningRateScheduler(lr_scheduler)
#
#net = vgg16_bn()
#net.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#net.fit_generator(
#    datagen_train.flow(images_train, labels_train, batch_size=BATCH_SZIE),
#    steps_per_epoch=len(images_train) / 32,
#    epochs=160,
#    callbacks=[tb, lr_sch]
#)
#
#net.save('checkpoint/vgg16_bn.h5')
#
#
#
#


def train(dataloader, train):
    simclr


def main():

    dataset = torchvision.datasets.STL10('data', split='unlabeled', download=True, transform=simclr_transforms)
    print(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2**5)
    #dataloader = torch.

    # We use ResNet-50 as the base encoder net- work, 
    # and a 2-layer MLP projection head to project the 
    # representation to a 128-dimensional latent space.
    net = alexnet(128)

    simclr = SimCLR(net)
    #print(simclr)
    
    




if __name__ == '__main__':
    main()