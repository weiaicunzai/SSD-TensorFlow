"""Define a vgg16_bn model
using tf.keras
"""

from tensorflow.python import keras


def vgg16_bn(num_class=100):
    model = keras.models.Sequential()

    #Conv1
    model.add(keras.layers.Conv2D(64, 3, padding='same', 
                                  input_shape=([32, 32, 3]),
                                  kernel_initializer='he_normal'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Conv2D(64, 3, padding='same',
                                  kernel_initializer='he_normal'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.MaxPool2D(strides=2))

    #Conv2
    model.add(keras.layers.Conv2D(128, 3, padding='same',
                                  kernel_initializer='he_normal'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Conv2D(128, 3, padding='same',
                                  kernel_initializer='he_normal'))
    model.add(keras.layers.BatchNormalization())                    
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.MaxPool2D(strides=2))

    #Conv3
    model.add(keras.layers.Conv2D(256, 3, padding='same',
                                  kernel_initializer='he_normal'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Conv2D(256, 3, padding='same',
                                  kernel_initializer='he_normal'))
    model.add(keras.layers.BatchNormalization())                            
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Conv2D(256, 3, padding='same',
                                  kernel_initializer='he_normal'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.MaxPool2D(strides=2))

    #Conv4
    model.add(keras.layers.Conv2D(512, 3, padding='same',
                                  kernel_initializer='he_normal'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Conv2D(512, 3, padding='same',
                                  kernel_initializer='he_normal'))
    model.add(keras.layers.BatchNormalization()) 
    model.add(keras.layers.Activation('relu'))
    
    model.add(keras.layers.Conv2D(512, 3, padding='same',
                                  kernel_initializer='he_normal'))
    model.add(keras.layers.BatchNormalization())                                
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.MaxPool2D(strides=2))

    #Conv5
    model.add(keras.layers.Conv2D(512, 3, padding='same',
                                  kernel_initializer='he_normal'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Conv2D(512, 3, padding='same',
                                  kernel_initializer='he_normal'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Conv2D(512, 3, padding='same',
                                  kernel_initializer='he_normal'))
    model.add(keras.layers.BatchNormalization())                                
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.MaxPool2D(strides=2))

    model.add(keras.layers.Flatten())

    #Fc1
    model.add(keras.layers.Dense(4096))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.5))

    #Fc2
    model.add(keras.layers.Dense(4096))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.5))

    #Fc3
    model.add(keras.layers.Dense(num_class))
    model.add(keras.layers.Activation('softmax'))

    return model

vgg16_bn()