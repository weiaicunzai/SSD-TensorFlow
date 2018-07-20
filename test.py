from vgg16_bn import vgg16_bn 
from dataset import *
from tensorflow.python import keras




model = keras.models.load_model('checkpoint/vgg16_bn.h5')
print(model)
datagen_test = keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    featurewise_center=True,
    rotation_range=15,
    width_shift_range=4,
    featurewise_std_normalization=True,
)

images_test, labels_test = cifar100_test('/home/admin-bai/Downloads/cifar-100-python')
labels_test = keras.utils.to_categorical(labels_test, 100)
datagen_test.fit(images_test)

