import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
# x_image = tf.reshape(x, [-1, 28, 28, 1])

# 定义权重函数
def weight_variables(shape):
    initial = tf.truncated_normal(shape)
    return tf.Variable(initial)


# 定义偏置函数
def bias_variables(shape):
    initial = tf.truncated_normal(shape)
    return tf.Variable(initial)


# 定义卷积运算函数
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")


# 定义池化运算函数
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def LeNet_forward1(x_image):
    x_image = tf.reshape(x_image, [-1, 28, 28, 1])
    # '''
    # 定义第一卷积层和第一池化层
    w_conv1 = weight_variables([5, 5, 1, 6])  # 注意这里卷积核的个数为6个 filter:[height,weight,in_channels,out_channels]
    b_conv1 = bias_variables([6])
    # 第一层卷积输出
    conv1 = conv2d(x_image, w_conv1)  # x_image:[-1,28,28,1] w_conv1: filter
    h_conv1 = tf.nn.sigmoid(tf.add(conv1, b_conv1))  # 卷积加偏置，经过激活函数输出，此为卷积层的输出
    # 第二层池化输出
    h_pool1 = max_pool_2x2(h_conv1)  # 此为池化层的输出

    # 定义第二卷积层和第二池化层
    w_conv2 = weight_variables([5, 5, 6, 16])  # 注意这里卷积核的个数为16个
    b_conv2 = bias_variables([16])
    # 第三层卷积输出
    conv2 = conv2d(h_pool1, w_conv2)
    h_conv2 = tf.nn.sigmoid(tf.add(conv2, b_conv2))
    # 第四层池化输出
    h_pool2 = max_pool_2x2(h_conv2)
    # 定义第三卷积层，注意这次没有池化层，而是直接连接全连接层
    w_conv3 = weight_variables([5, 5, 16, 120])  # 卷积核的个数为120个
    b_conv3 = bias_variables([120])
    # 第五层卷积输出，输出的shape为[?,7,7,120]
    conv3 = conv2d(h_pool2, w_conv3)
    h_conv3 = tf.nn.sigmoid(tf.add(conv3, b_conv3))

    # 定义第六层全连接层
    w_fc1 = weight_variables([7 * 7 * 120, 80])
    b_fc1 = bias_variables([80])
    # 把即将进入全连接层的输入h_conv3重塑为一维向量
    h_conv3_flat = tf.reshape(h_conv3, [-1, 7 * 7 * 120])
    # 第六层全连接层的输出
    h_fc1 = tf.nn.sigmoid(tf.add(tf.matmul(h_conv3_flat, w_fc1), b_fc1))
    # logit = h_fc1
    # 最后一层全连接层，使用softmax进行分类
    w_fc2 = weight_variables([80, 10])
    b_fc2 = bias_variables([10])
    y_model = tf.nn.softmax(tf.add(tf.matmul(h_fc1, w_fc2), b_fc2))

    # 损失函数，采用交叉熵
    # loss = -tf.reduce_sum(y_ * tf.log(y_model))

    logit = y_model

    return logit
#
#
# if __name__ == '__main__':
#     tf.app.run()

def change_to_right(wrong_labels):
    right_labels=[]
    for x in wrong_labels:
        for i in range(0,len(wrong_labels[0])):
            if x[i]==1:
                right_labels.append(i-1)
    return right_labels

def change_to_right1(label):
    right_labels=[]
    i = 0
    for x in label:
        i += 1
        if x == 1:
            right_labels.append(i-1)
    return np.array(right_labels)

import tensorflow as tf
from tensorflow.contrib.slim.nets import vgg
import numpy as np
import foolbox

images = tf.placeholder(tf.float32, shape=(None, 784))
# preprocessed = images - [123.68, 116.78, 103.94]
logits = LeNet_forward1(images)
restorer = tf.train.Saver(tf.trainable_variables())

image, _ = foolbox.utils.imagenet_example()
#load MNIST dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist.test.images.shape)
image = np.array(mnist.test.images[25])
label= np.array(mnist.test.labels[25])
# print("-->image = mnist.test.images[0]:")
# print(len(image), image.shape, type(image))
# print(len(label), label.shape, type(label))

true_label = change_to_right1(label)
# print(true_label, type(true_label), len(true_label), true_label.shape)
print(true_label)

true_image =  tf.reshape(image, [1,784])
# true_image = true_image.numpy()
sess = tf.InteractiveSession()
true_image = true_image.eval()
# print(true_image)
# print(true_image, type(true_image), len(true_image), true_image.shape)


with foolbox.models.TensorFlowModel(images, logits, (0, 1)) as model:
    restorer.restore(model.session, '/Users/apple/Desktop/foolbox/foolbox/MNIST_model/LeNet_model.ckpt')
    print(image.shape)
    print(np.argmax(model.forward_one(image)))

images, labels = foolbox.utils.samples(dataset='mnist', batchsize=20, data_format='channels_first', bounds=(0, 1))
true_images = tf.reshape(images, [20, 784])
sess = tf.InteractiveSession()
true_images = true_images.eval()
print(labels)
true_labels = tf.reshape(labels, [20, ])
sess = tf.InteractiveSession()
true_labels = true_labels.eval()
print(np.mean(model.forward(true_images).argmax(axis=-1) == true_labels))

attack = foolbox.attacks.FGSM (model)
adversarials = attack(true_image, true_label)
adversal = adversarials[0]
print(adversal.shape)
print(np.argmax(model.forward_one(adversal)))
print("-->adversarials:")
print(np.mean(model.forward(adversarials).argmax(axis=-1) == true_label))
#--> 0.0

print(np.mean(model.forward(true_image).argmax(axis=-1) == true_label))
#-->1.0



print("是否改变图片？：")
print(adversarials == true_image)
#all changes??????????????? (全部False）


import matplotlib.pyplot as plt

attack_image = tf.reshape(adversarials, [28,28])
sess = tf.InteractiveSession()
attack_image = attack_image.eval()

true_image = tf.reshape(true_image, [28,28])
sess = tf.InteractiveSession()
true_image = true_image.eval()
plt.imshow(true_image) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()

plt.imshow(attack_image) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()

print("-->true image:")
print(true_image)
print("-->adversarial:")
print(adversarials)
