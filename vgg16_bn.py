"""Define a vgg16_bn model
using tf.contrib.slim
"""

import tensorflow as tf
slim = tf.contrib.slim

def vgg16_bn(inputs, num_class=100, scope='vgg16_bn'):
    with tf.variable_scope(scope, 'vgg16_bn', [inputs]) as sc:

        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 1, slim.conv2d, 64, [3, 3], scope='conv1')
            print(net)

vgg16_bn(tf.ones([12, 32, 32, 3]))
#    model = keras.models.Sequential()
#
#    #Conv1
#    model.add(keras.layers.Conv2D(64, 3, padding='same', 
#                                  input_shape=([32, 32, 3]),
#                                  kernel_initializer='he_normal'))
#    model.add(keras.layers.BatchNormalization())
#    model.add(keras.layers.Activation('relu'))
#
#    model.add(keras.layers.Conv2D(64, 3, padding='same',
#                                  kernel_initializer='he_normal'))
#    model.add(keras.layers.BatchNormalization())
#    model.add(keras.layers.Activation('relu'))
#
#    model.add(keras.layers.MaxPool2D(strides=2))
#
#    #Conv2
#    model.add(keras.layers.Conv2D(128, 3, padding='same',
#                                  kernel_initializer='he_normal'))
#    model.add(keras.layers.BatchNormalization())
#    model.add(keras.layers.Activation('relu'))
#
#    model.add(keras.layers.Conv2D(128, 3, padding='same',
#                                  kernel_initializer='he_normal'))
#    model.add(keras.layers.BatchNormalization())                    
#    model.add(keras.layers.Activation('relu'))
#
#    model.add(keras.layers.MaxPool2D(strides=2))
#
#    #Conv3
#    model.add(keras.layers.Conv2D(256, 3, padding='same',
#                                  kernel_initializer='he_normal'))
#    model.add(keras.layers.BatchNormalization())
#    model.add(keras.layers.Activation('relu'))
#
#    model.add(keras.layers.Conv2D(256, 3, padding='same',
#                                  kernel_initializer='he_normal'))
#    model.add(keras.layers.BatchNormalization())                            
#    model.add(keras.layers.Activation('relu'))
#
#    model.add(keras.layers.Conv2D(256, 3, padding='same',
#                                  kernel_initializer='he_normal'))
#    model.add(keras.layers.BatchNormalization())
#    model.add(keras.layers.Activation('relu'))
#
#    model.add(keras.layers.MaxPool2D(strides=2))
#
#    #Conv4
#    model.add(keras.layers.Conv2D(512, 3, padding='same',
#                                  kernel_initializer='he_normal'))
#    model.add(keras.layers.BatchNormalization())
#    model.add(keras.layers.Activation('relu'))
#
#    model.add(keras.layers.Conv2D(512, 3, padding='same',
#                                  kernel_initializer='he_normal'))
#    model.add(keras.layers.BatchNormalization()) 
#    model.add(keras.layers.Activation('relu'))
#    
#    model.add(keras.layers.Conv2D(512, 3, padding='same',
#                                  kernel_initializer='he_normal'))
#    model.add(keras.layers.BatchNormalization())                                
#    model.add(keras.layers.Activation('relu'))
#
#    model.add(keras.layers.MaxPool2D(strides=2))
#
#    #Conv5
#    model.add(keras.layers.Conv2D(512, 3, padding='same',
#                                  kernel_initializer='he_normal'))
#    model.add(keras.layers.BatchNormalization())
#    model.add(keras.layers.Activation('relu'))
#
#    model.add(keras.layers.Conv2D(512, 3, padding='same',
#                                  kernel_initializer='he_normal'))
#    model.add(keras.layers.BatchNormalization())
#    model.add(keras.layers.Activation('relu'))
#
#    model.add(keras.layers.Conv2D(512, 3, padding='same',
#                                  kernel_initializer='he_normal'))
#    model.add(keras.layers.BatchNormalization())                                
#    model.add(keras.layers.Activation('relu'))
#
#    model.add(keras.layers.MaxPool2D(strides=2))
#
#    model.add(keras.layers.Flatten())
#
#    #Fc1
#    model.add(keras.layers.Dense(4096))
#    model.add(keras.layers.Activation('relu'))
#    model.add(keras.layers.Dropout(0.5))
#
#    #Fc2
#    model.add(keras.layers.Dense(4096))
#    model.add(keras.layers.Activation('relu'))
#    model.add(keras.layers.Dropout(0.5))
#
#    #Fc3
#    model.add(keras.layers.Dense(num_class))
#    model.add(keras.layers.Activation('softmax'))
#
#    return model
#
#vgg16_bn()