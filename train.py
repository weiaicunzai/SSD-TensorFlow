import tensorflow as tf
import dataset as ds
from vgg16_bn import vgg16_bn 



sess = tf.Session()
BATCH_SZIE = 32 
CIFAR100 = "/home/admin-bai/Downloads/cifar-100-python"

def train():

    def train_input_fn():
        dataset = ds.cifar100_train(CIFAR100, BATCH_SZIE)
        data_iterater = dataset.make_one_shot_iterator()
        images, labels = data_iterater.get_next()
        return images, labels

    

    cifar100_classifier = tf.estimator.Estimator(
        model_fn=vgg16_bn, model_dir="checkpoint"
    )

    cifar100_classifier.train(
        input_fn=train_input_fn,
        steps=float(50000 / BATCH_SZIE)
    )



train()


