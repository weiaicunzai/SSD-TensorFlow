import os
from datetime import datetime

from simclr import SimCLR
from model import alexnet
from transform import simclr_transforms
from loss import InfoNCE

import torch
import torch.nn as nn
import torchvision
import torchmetrics

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
#def eval(dataloader, model):
#    infonce_loss = InfoNCE()
#    ce_loss = nn.CrossEntropyLoss()
#    metric = torchmetrics.Accuracy().cuda()
#
#
#    total_loss = 0
#    for images, _ in dataloader:
#        images = torch.cat(images, dim=0).cuda()
#        with torch.no_grad():
#            features = model(images)
#            logits, labels = infonce_loss(features)
#            loss = ce_loss(logits, labels)
#            total_loss += loss
#            print(logits.device, features.device)
#            metric.update(logits.softmax(dim=-1), labels)
#
#
#        print(total_loss / len(dataloader.dataset))
#
#        acc = metric.compute()
#        print(acc)
#
#
#    return acc
#
            

def train(epochs, dataloader, model, optimizer, lr_scheduler):

    DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
    TIME_NOW = datetime.now().strftime(DATE_FORMAT)
    save_path = os.path.join('checkpoint', TIME_NOW)

    infonce_loss = InfoNCE()
    ce_loss = nn.CrossEntropyLoss()


    best_acc = 0
    for epoch in range(epochs):
        metric = torchmetrics.Accuracy().cuda()
        for images, _ in dataloader:
            #print(images.shape)
            images = torch.cat(images, dim=0).cuda()
            print(images.shape)
            # print(images.shape)
            # print(images.shape)
            # count += images.shape[0]

            # finish = time.time()

            # print(count / (finish - start))
            features = model(images)
            logits, labels = infonce_loss(features)
            loss = ce_loss(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print('{:04f}'.format(loss.item()))

            metric.update(logits.softmax(dim=-1), labels)


        #acc = eval(dataloader, model)
        #print(acc)
        acc = metric.compute()
        print(acc)
        print()

        if best_acc < acc:
            best_acc = acc
            #if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, '{}.pth'.format(epoch)))





def main():
    lr = 0.0003
    epochs = 100

    #print(path)
    #import sys; sys.exit()

    dataset = torchvision.datasets.STL10('/data/ssd1/baiyu/data', split='unlabeled', download=True, transform=simclr_transforms(96))
    # print(simclr_transforms)
    # print(torchvision.datasets.__file__)
    # print(dataset[0])
    


    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2**10, num_workers=8)

    # We use ResNet-50 as the base encoder net- work, 
    # and a 2-layer MLP projection head to project the 
    # representation to a 128-dimensional latent space.
    net = alexnet(128)

    simclr = SimCLR(net).cuda()
    # print(simclr)
    optimizer = torch.optim.Adam(simclr.parameters(), lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader), eta_min=0,
                                                           last_epoch=-1)


    train(epochs, dataloader, simclr, optimizer, lr_scheduler)

    




if __name__ == '__main__':
    main()
    #metric = torchmetrics.Accuracy()
    #predict = torch.tensor([[0.1, 0.4, 0.2, 0.2, 0.1]])
    #labels = torch.tensor([1])
    ##acc = metric(predict, labels)
    ##acc = metric(predict, labels)
    ##acc = metric(predict, labels)
    #metric.update(predict, labels)
    #predict = torch.tensor([[0.1, 0.4, 0.2, 0.2, 0.1]])
    #labels = torch.tensor([2])
    #metric.update(predict, labels)
    #acc = metric.compute()
    #print(acc)