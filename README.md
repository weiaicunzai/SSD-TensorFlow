# TensorFlow-VGG

This is a TensorFlow version implemenation of VGG network, the purpose of this
repo is to pratice how to use TensorFlow.

## Abstruct
I was going to use TensorFlow's low level api to implement this VGG network, but
after I looking into the low level api, I've found that low level api is not that
easy to use, therefore I decided to use TensorFlow's high level api tensorflow.python.keras
as my main high level TensorFlow api for daily development. As for the low level api, 
my goal now is to simply understand what these low apis do.

## Requirement
TensorFlow 1.4
Python 3.5

## Things that I want to know about tf.keras API

- Data preprocessing
- How to Train a model
- How to use Tensorboard to debug network(basic functions)

# Conclusion
I've decided quit using TensorFlow for my development of Neuron Network, even the tf.Keras
api, I will switch to Keras api, and using Tensorflow as backend.Cause The tf.keras api is
slightly different than the standard Keras api, for example:
tf.kerar:
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
```
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

```
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

```
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
Therefore, I decide to switch to standard Keras API(version 2.15) which was competitable with 
TensorFLow1.4.Maybe TensorFlow would have a better high level api in the future, util then, I'll
stick with Keras.