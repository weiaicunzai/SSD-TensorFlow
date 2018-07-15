import numpy as np
import tensorflow as tf

class VGG16:

    def __init__(self, class_nums=100):

        initializer = tf.initializers.random_normal

        #conv1
        self.conv1_1 = tf.layers.Conv2D(64, 3, padding='same', kernel_initializer=initializer)
        self.bn1_1 = tf.layers.BatchNormalization()

        self.conv1_2 = tf.layers.Conv2D(64, 3, padding='same', kernel_initializer=initializer)
        self.bn1_2 = tf.layers.BatchNormalization()

        self.pool1 = tf.layers.MaxPooling2D(2, 2)

        #conv2
        self.conv2_1 = tf.layers.Conv2D(128, 3, padding='same', kernel_initializer=initializer)
        self.bn2_1 = tf.layers.BatchNormalization()

        self.conv2_2 = tf.layers.Conv2D(128, 3, padding='same', kernel_initializer=initializer)
        self.bn2_2 = tf.layers.BatchNormalization()

        self.pool2 = tf.layers.MaxPooling2D(2, 2)

        #conv3
        self.conv3_1 = tf.layers.Conv2D(256, 3, padding='same', kernel_constraint=initializer)
        self.bn3_1 = tf.layers.BatchNormalization()

        self.conv3_2 = tf.layers.Conv2D(256, 3, padding='same', kernel_constraint=initializer)
        self.bn3_2 = tf.layers.BatchNormalization()

        self.conv3_3 = tf.layers.Conv2D(256, 3, padding='same', kernel_initializer=initializer)
        self.bn3_3 = tf.layers.BatchNormalization()

        self.pool3 = tf.layers.MaxPooling2D(2, 2)

        #conv4
        self.conv4_1 = tf.layers.Conv2D(512, 3, padding='same', kernel_initializer=initializer)
        self.bn4_1 = tf.layers.BatchNormalization()

        self.conv4_2 = tf.layers.Conv2D(512, 3, padding='same', kernel_initializer=initializer)
        self.bn4_2 = tf.layers.BatchNormalization()

        self.conv4_3 = tf.layers.Conv2D(512, 3, padding='same', kernel_initializer=initializer)
        self.bn4_3 = tf.layers.BatchNormalization()

        self.pool4 = tf.layers.MaxPooling2D(2, 2)

        #conv5
        self.conv5_1 = tf.layers.Conv2D(512, 3, padding='same', kernel_initializer=initializer)
        self.bn5_1 = tf.layers.BatchNormalization()

        self.conv5_2 = tf.layers.Conv2D(512, 3, padding='same', kernel_initializer=initializer)
        self.bn5_2 = tf.layers.BatchNormalization()

        self.conv5_3 = tf.layers.Conv2D(512, 3, padding='same', kernel_initializer=initializer)
        self.bn_5_3 = tf.layers.BatchNormalization()

        self.pool5 = tf.layers.MaxPooling2D(2, 2)

    def predict(self, x):
        with tf.name_scope("vgg16_bn"):

            #conv1
            with tf.name_scope('conv1'):
                x = self.conv1_1(x)
                x = self.bn1_1(x)
                x = tf.nn.relu(x)

                x = self.conv1_2(x)
                x = self.bn2_1(x)
                x = tf.nn.relu(x)

                x = self.pool1(x)
            
            #conv2
            with tf.name_scope('conv2'):
                x = self.conv2_1(x)
                x = self.bn2_1(x)
                x = tf.nn.relu(x)

                x = self.conv2_2(x)
                x = self.bn2_2(x)
                x = tf.nn.relu(x)

                x = self.pool2(x)
            
            #conv3
            with tf.name_scope('conv3'):
                x = self.conv3_1(x)
                x = self.bn3_1(x)
                x = tf.nn.relu(x)

                x = self.conv3_2(x)
                x = self.bn3_2(x)
                x = tf.nn.relu(x)

                x = self.conv3_3(x)
                x = self.bn3_3(x)
                x = tf.nn.relu(x)

                x = self.pool3(x)
            
            #conv4
            with tf.name_scope('conv4'):
                x = self.conv4_1(x)
                x = self.bn4_1(x)
                x = tf.nn.relu(x)

                x = self.conv4_2(x)
                x = self.bn4_2(x)
                x = tf.nn.relu(x)

                x = self.conv4_3(x)
                x = self.bn4_3(x)
                x = tf.nn.relu(x)

                x = self.pool4(x)
            
            #conv5
            with tf.name_scope('conv5'):
                x = self.conv5_1(x)
                x = self.bn5_1(x)
                x = tf.nn.relu(x)

                x = self.conv5_2(x)
                x = self.bn5_2(x)
                x = tf.nn.relu(x)

                x = self.conv5_3(x)
                x = self.bn5_3(x)
                x = tf.nn.relu(x)

                x = self.pool5(x)


