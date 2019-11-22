import foolbox
import numpy as np
import torchvision.models as models
# 全局取消证书验证
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import sys
import input_data

# Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
tf.logging.set_verbosity(tf.logging.INFO)
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.training import coordinator
import time

#load MNIST dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("testing data size:", mnist.test.num_examples)
train_data = mnist.train.images  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images  # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

#验证是否load成功
# index = 7
# plt.imshow(train_data[index].reshape(28, 28))
# plt.show()
# print ("y = " + str(np.squeeze(train_labels[index])))

print ("number of training examples = " + str(train_data.shape[0]))
print ("number of evaluation examples = " + str(eval_data.shape[0]))
print ("X_train shape: " + str(train_data.shape))
print ("Y_train shape: " + str(train_labels.shape))
print ("X_test shape: " + str(eval_data.shape))
print ("Y_test shape: " + str(eval_labels.shape))

def change_to_right(wrong_labels):
    right_labels=[]
    for x in wrong_labels:
        for i in range(0,len(wrong_labels[0])):
            if x[i]==1:
                right_labels.append(i)
    return right_labels

'''
def cnn_model_fn(features, labels, mode):
    print("-->the start of creating estimator!")
    # Input Layer
    input_height, input_width = 28, 28
    input_channels = 1
    input_layer = tf.reshape(features["x"], [-1, input_height, input_width, input_channels])

    # Convolutional Layer #1 and Pooling Layer #1
    conv1_1 = tf.layers.conv2d(inputs=input_layer, filters=64, kernel_size=[3, 3], padding="same",
                               activation=tf.nn.relu)
    conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=[2, 2], strides=2, padding="same")

    # Convolutional Layer #2 and Pooling Layer #2
    conv2_1 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[2, 2], strides=2, padding="same")

    # Convolutional Layer #3 and Pooling Layer #3
    conv3_1 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3_2, pool_size=[2, 2], strides=2, padding="same")

    # Convolutional Layer #4 and Pooling Layer #4
    conv4_1 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    conv4_2 = tf.layers.conv2d(inputs=conv4_1, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    pool4 = tf.layers.max_pooling2d(inputs=conv4_2, pool_size=[2, 2], strides=2, padding="same")

    # Convolutional Layer #5 and Pooling Layer #5
    conv5_1 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    conv5_2 = tf.layers.conv2d(inputs=conv5_1, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    pool5 = tf.layers.max_pooling2d(inputs=conv5_2, pool_size=[2, 2], strides=2, padding="same")

    # FC Layers
    pool5_flat = tf.contrib.layers.flatten(pool5)
    FC1 = tf.layers.dense(inputs=pool5_flat, units=4096, activation=tf.nn.relu)
    FC2 = tf.layers.dense(inputs=FC1, units=4096, activation=tf.nn.relu)
    FC3 = tf.layers.dense(inputs=FC2, units=1000, activation=tf.nn.relu)

    """the training argument takes a boolean specifying whether or not the model is currently 
    being run in training mode; dropout will only be performed if training is true. here, 
    we check if the mode passed to our model function cnn_model_fn is train mode. """
    dropout = tf.layers.dropout(inputs=FC3, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer or the output layer. which will return the raw values for our predictions.
    # Like FC layer, logits layer is another dense layer. We leave the activation function empty
    # so we can apply the softmax
    logits = tf.layers.dense(inputs=dropout, units=10)

    # Then we make predictions based on raw output
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        # the predicted class for each example - a vlaue from 0-9
        "classes": tf.argmax(input=logits, axis=1),
        # to calculate the probablities for each target class we use the softmax
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # so now our predictions are compiled in a dict object in python and using that we return an estimator object
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # print("type of labels:", type(labels))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess, coord)
    # labels1 = labels.eval(session=sess)
    # labels1 = np.array(change_to_right(labels))
    # labels1 = tf.convert_to_tensor(np.array(labels))
    # labels1 = tf.convert_to_tensor(np.array(change_to_right(labels)))
    labels1 = labels.eval(session=sess)
    # print("type of labels1:", type(labels1))
    # print("labels1:")
    # print(labels1)
    labels1 = change_to_right(labels1)
    # print("labels1: with len", len(labels1))
    # print(labels1)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels1, logits=logits)

    # Configure the Training Options (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        print("-->the end of creating estimator!(in if)")
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels,
                                        predictions=predictions["classes"])}
    print("-->the end of creating estimator!")
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops)


#create an estimator object and name it mnist_classifier
print("-->create an estimator!")
mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                          model_dir="/tmp/mnist_vgg13_model")

time.sleep(5)

#define training parameters and hyperparameters
print("-->define training parameters and hyperparameters!")
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data},
                                                        y=train_labels,
                                                        batch_size=100,
                                                        num_epochs=100,
                                                        shuffle=True)

# train_input_fn = numpy_io.numpy_input_fn(x={"x": train_data},
#                                          y=change_to_right(train_labels),
#                                          batch_size=100,
#                                          num_epochs=100,
#                                          shuffle=True)

# mnist_classifier.train(input_fn=train_input_fn,
#                        steps=None,
#                        hooks=None)

print("-->training model!")
mnist_classifier.train(input_fn=train_input_fn)
print("-->the end of training!")


print("-->set testing data!")
eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data},
                                                   y=eval_labels,
                                                   num_epochs=1,
                                                   shuffle=False)
print("-->testing model!")
eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)
'''

# instantiate model (supports PyTorch, Keras, TensorFlow (Graph and Eager), JAX, MXNet and many more)
#torchvision.model #pretrained If True, returns a model pre-trained on ImageNet
# model = models.resnet18(pretrained=True).eval()
model = models.resnet18(pretrained=True).eval()
#normalize = transforms.Normalize #expect input images normalized
preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=1000, preprocessing=preprocessing)

# get a batch of images and labels and print the accuracy
images, labels = foolbox.utils.samples(dataset='imagenet', batchsize=16, data_format='channels_first', bounds=(0, 1))
print(np.mean(fmodel.forward(images).argmax(axis=-1) == labels))
# -> 0.9375
# -> 0.875 (vgg11) -> 0.6875 (vgg11_bn)  -> 0.875 (vgg19) ->  0,8125 (vgg16)

#16 images and 16 labels

image = np.array([images[0,...]])
# print(image, len(image))
label = np.array([labels[0]])
# print(label, len(label))

# apply the attack
# attack = foolbox.attacks.BoundaryAttack(fmodel)
attack = foolbox.attacks.BoundaryAttack(fmodel)
# adversarials = attack(images, labels) #adversarials have no label
adversarials = attack(image, label)
# print(adversarials, len(adversarials))
# if the i'th image is misclassfied without a perturbation, then adversarials[i] will be the same as images[i]
# if the attack fails to find an adversarial for the i'th image, then adversarials[i] will all be np.nan

# Foolbox guarantees that all returned adversarials are in fact in adversarials
print(np.mean(fmodel.forward(adversarials).argmax(axis=-1) == labels))
# print(np.mean(fmodel.forward(adversarials).argmax(axis=-1) == label))
# -> 0.0
