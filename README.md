# SSD-TensorFlow

This is a TensorFlow version implemenation of [SSD(Single Shot MultiBox Detector)](https://arxiv.org/abs/1512.02325v5)
network, the purpose of this repo is to pratice how to use TensorFlow.

## Abstruct
I was going to use TensorFlow's low level api to implement this SSD, but
after I looking into the low level api, I've found that low level api is not that
easy to use, therefore I decided to use TensorFlow's high level api tf.contrib.slim
as my main high level TensorFlow api for development. As for the low level api, 
my goal now is to simply understand what these low apis do.

## Requirement
TensorFlow 1.4
Python 3.5

# Notations:

**tf.contrib will be deprecated**
according to (this google groups link)[https://groups.google.com/a/tensorflow.org/forum/m/#!msg/announce/qXfsxr2sF-0/jHQ77dr3DAAJ]

```
TensorFlow’s contrib module has grown beyond what can be maintained and supported in a single repository. Larger projects are better maintained separately, while we will incubate smaller extensions along with the main TensorFlow code. Consequently, as part of releasing TensorFlow 2.0, we will stop distributing tf.contrib. We will work with the respective owners on detailed migration plans in the coming months, including how to publicise your TensorFlow extension in our community pages and documentation. For each of the contrib modules we will either 
a) integrate the project into TensorFlow; 
b) move it to a separate repository or 
c) remove it entirely. 
This does mean that all of tf.contrib will be deprecated, and we will stop adding new tf.contrib projects today. We are looking for owners/maintainers for a number of projects currently in tf.contrib, please contact us (reply to this email) if you are interested. 
```

In other words, tf.contrib.slim could be removed in tf 2.0, we dont know for sure.

I will be using tf.contrib.slim heigh level api to write the code, 
the reason not to use tf.Keras are as follows

in tf.kerar:
Class ImageDataGenerator
```python
__init__(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=0.0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    shear_range=0.0,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode='nearest',
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format=None
)
```

I can't use validation_split argument in validation_split

But when it comes to Keras 2.1.5, in their release note:

```
Add validation_split API in keras.preprocessing.image.ImageDataGenerator. 
You can pass validation_split to the constructor (float), then select between 
training/validation subsets by passing the argument subset='validation' or 
subset='training' to methods flow and flow_from_directory.
```

So I installed Keras2.1.5, and using the same version if TensorFlow which
is 1.4.0, the following code can run without any exceptions:
```python
datagen_train = ImageDataGenerator(
    horizontal_flip=True,
    featurewise_center=True,
    rotation_range=15,
    width_shift_range=4,
    featurewise_std_normalization=True,
    validation_split=0.1
)
``` 

I dont want to upgrade my TensorFlow to higher version since it requires to 
update my CUDA8.0 to CUDA9.0, but other programs in my machine aslo need CUDA8.0.


And another example of me decide to stop using TensorFlow as my main deep learning
framework is this:

```python
tb = keras.callbacks.TensorBoard(log_dir='log')
lr_sch = keras.callbacks.LearningRateScheduler(lr_scheduler)

net = vgg16_bn()
net.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
net.fit_generator(
    datagen_train.flow(images_train, labels_train, batch_size=BATCH_SZIE),
    steps_per_epoch=len(images_train) / 32,
    epochs=4,
    callbacks=[tb, lr_sch]
)
```

if I want to use keras.callbacks.TensorBoard to draw histgrams, I need to specify the 
validation datset in fit_generator() method, but in TensorFlow1.4, fit_generator() method
don't support generator as validation dataset, which means I have to manually preprocess
the validation dataset.

```python
net.fit_generator(
    datagen_train.flow(images_train, labels_train, batch_size=BATCH_SZIE),
    steps_per_epoch=len(images_train) / 32,
    epochs=4,
    callbacks=[tb, lr_sch],
    validation_data=datagen_test.flow(images_test, labels_test,   #Error, doesn't support generator 
            batch_size=BATCH_SIZE) 
)
```

Maybe there is another workaround, but I am tired of tring, this is not the only problem I've 
encountered that was caused by the mess of TensorFlow high level api.
Therefore, Standard Keras API(version 2.15) is a better choice which was competitable with 
TensorFLow1.4. So tf.keras is not suitable for me.