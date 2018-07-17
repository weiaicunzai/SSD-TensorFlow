import numpy as np
import tensorflow as tf


def vgg16_bn(features, labels, mode):

        class_nums = 100
        initializer = tf.initializers.random_normal

        conv1_1 = tf.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer=initializer)
        bn1_1 = tf.layers.BatchNormalization()

        conv1_2 = tf.layers.Conv2D(64, 3, padding='same', kernel_initializer=initializer)
        bn1_2 = tf.layers.BatchNormalization()

        pool1 = tf.layers.MaxPooling2D(2, 2)

        conv2_1 = tf.layers.Conv2D(128, 3, padding='same', kernel_initializer=initializer)
        bn2_1 = tf.layers.BatchNormalization()

        conv2_2 = tf.layers.Conv2D(128, 3, padding='same', kernel_initializer=initializer)
        bn2_2 = tf.layers.BatchNormalization()

        pool2 = tf.layers.MaxPooling2D(2, 2)

        conv3_1 = tf.layers.Conv2D(256, 3, padding='same', kernel_constraint=initializer)
        bn3_1 = tf.layers.BatchNormalization()

        conv3_2 = tf.layers.Conv2D(256, 3, padding='same', kernel_constraint=initializer)
        bn3_2 = tf.layers.BatchNormalization()

        conv3_3 = tf.layers.Conv2D(256, 3, padding='same', kernel_initializer=initializer)
        bn3_3 = tf.layers.BatchNormalization()

        pool3 = tf.layers.MaxPooling2D(2, 2)

        conv4_1 = tf.layers.Conv2D(512, 3, padding='same', kernel_initializer=initializer)
        bn4_1 = tf.layers.BatchNormalization()

        conv4_2 = tf.layers.Conv2D(512, 3, padding='same', kernel_initializer=initializer)
        bn4_2 = tf.layers.BatchNormalization()

        conv4_3 = tf.layers.Conv2D(512, 3, padding='same', kernel_initializer=initializer)
        bn4_3 = tf.layers.BatchNormalization()

        pool4 = tf.layers.MaxPooling2D(2, 2)

        conv5_1 = tf.layers.Conv2D(512, 3, padding='same', kernel_initializer=initializer)
        bn5_1 = tf.layers.BatchNormalization()

        conv5_2 = tf.layers.Conv2D(512, 3, padding='same', kernel_initializer=initializer)
        bn5_2 = tf.layers.BatchNormalization()

        conv5_3 = tf.layers.Conv2D(512, 3, padding='same', kernel_initializer=initializer)
        bn5_3 = tf.layers.BatchNormalization()

        pool5 = tf.layers.MaxPooling2D(2, 2)

        flatten = tf.layers.Flatten()
        fc6 = tf.layers.Dense(512, activation=tf.nn.relu, kernel_initializer=initializer)
        droptout = tf.layers.Dropout()
        fc7 = tf.layers.Dense(4086, activation=tf.nn.relu, kernel_constraint=initializer)
        fc8 = tf.layers.Dense(class_nums, activation=tf.nn.relu, kernel_initializer=initializer)

        x = features

        x = tf.reshape(x, [-1, 32, 32, 3])

        with tf.name_scope("vgg16_bn"):
            #conv1
            with tf.name_scope('conv1'):
                #conv1 = tf.layers.conv2d(
                #    inputs=x,
                #    filters=32,
                #    kernel_size=[3, 3],
                #    padding="same",
                #    activation=tf.nn.relu)

                #conv1 = tf.layers.Conv2D(
                #    64,
                #    (3, 3),
                #    padding='same',
                #    activation=tf.nn.relu
                #)
                #x = conv1
                x = conv1_1(x)
                x = bn1_1(x)
                x = tf.nn.relu(x)

                x = conv1_2(x)
                x = bn1_2(x)
                x = tf.nn.relu(x)

                x = pool1(x)
            
            #conv2
            with tf.name_scope('conv2'):
                x = conv2_1(x)
                x = bn2_1(x)
                x = tf.nn.relu(x)

                x = conv2_2(x)
                x = bn2_2(x)
                x = tf.nn.relu(x)

                x = pool2(x)
            
            #conv3
            with tf.name_scope('conv3'):
                x = conv3_1(x)
                x = bn3_1(x)
                x = tf.nn.relu(x)

                x = conv3_2(x)
                x = bn3_2(x)
                x = tf.nn.relu(x)

                x = conv3_3(x)
                x = bn3_3(x)
                x = tf.nn.relu(x)

                x = pool3(x)
            
            #conv4
            with tf.name_scope('conv4'):
                x = conv4_1(x)
                x = bn4_1(x)
                x = tf.nn.relu(x)

                x = conv4_2(x)
                x = bn4_2(x)
                x = tf.nn.relu(x)

                x = conv4_3(x)
                x = bn4_3(x)
                x = tf.nn.relu(x)

                x = pool4(x)
            
            #conv5
            with tf.name_scope('conv5'):
                x = conv5_1(x)
                x = bn5_1(x)
                x = tf.nn.relu(x)

                x = conv5_2(x)
                x = bn5_2(x)
                x = tf.nn.relu(x)

                x = conv5_3(x)
                x = bn5_3(x)
                x = tf.nn.relu(x)

                x = pool5(x)

            with tf.name_scope('fc6'):
                x = flatten(x)
                x = fc6(x)
                x = droptout(x)
                x = tf.nn.relu(x)

                x = fc7(x)
                x = droptout(x)
                x = tf.nn.relu(x)

                x = fc8(x)
                x = tf.nn.relu(x)

            
            logits = x
            predictions = {
                    # Generate predictions (for PREDICT and EVAL mode)
                    "classes": tf.argmax(input=logits, axis=1),
                    # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
                    # `logging_hook`.
                    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }

            loss = tf.losses.softmax_cross_entropy(
                onehot_labels=labels, logits=logits
            )

            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
                train_op = optimizer.minimize(
                    loss=loss
                )

                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

            # Add evaluation metrics (for EVAL mode)
            eval_metric_ops = {
                "accuracy": tf.metrics.accuracy(
                    labels=labels, predictions=predictions["classes"])}
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

