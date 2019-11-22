"""
卷积神经网络处理的过程，主要包含4个步骤
1. 图像输入：获取输入的数据图像，一般需要经历float数据类型转换，4维Tensor转换
2. 卷积：对图像特征进行提取
3. maxpool：压缩聚合卷积提取到的特征，起到降维和改善结果的作用
4. 全连接层：用于对图像进行分类

卷积网络在本质上是一种输入到输出的映射，能够学习大量的输入与输出之间的映射关系，而
不需要任何输入与输出之间的精确的数学表达式

在开始训练前，所有的权重都应该用一些不同的小随机数进行初始化。“小随机数”用来保证
网络不会因为权重过大而进入饱和状态，从而导致训练失败。“不同”用来保证网络可以正常地
学习，实际上，如果用相同的数去初始化矩阵，则网络无学习能力
"""
# TensorFlow实现LeNet实例
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time  # 用来观察模型训练的时间
import matplotlib.pyplot as plt  # 用于绘制准确率曲线

import numpy as np

#load MNIST dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# print("testing data size:", mnist.test.num_examples)
# train_data = mnist.train.images  # Returns np.array
# train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
# eval_data = mnist.test.images  # Returns np.array
# eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

# 载入mnist数据集
# mnist = input_data.read_data_sets("tensorflow_application/MNIST_data/", one_hot=True)
# 定义占位符，声明输入图片的数据和类别及输出的数据和类别
#2维tensor none可以为任何batch size
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
# 对输入数据进行转化，需要转变为4维的Tensor用于卷积神经网络的输入
print(mnist.train.images.shape)  # mnist数据集是以[None, 784]的数据格式存放的
# print(type(mnist.train))
# print(type(mnist.train.images))
# print(mnist.train.images[0])
# print(type(mnist.train.labels))
# print(mnist.train.labels[0])
# print(type(mnist.train.images[0]))
# batch = mnist.train.next_batch(10)
# print(batch)
# print(type(batch))
# x = batch[0]
# y = batch[1]
# print("-->x")
# print(x, type(x))
# print("-->y")
# print(y, type(y))
# print("labels' type")
# print(type(mnist.train.labels))

train_data = np.asarray(mnist.train.images, dtype=np.float32)  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.float32)
eval_data = np.asarray(mnist.test.images, dtype=np.float32) # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.float32)

print(train_data.shape,train_labels.shape)
train_data1 = train_data[0:44000,:]
train_labels1 = train_labels[0:44000,:]
eval_data1 = eval_data[0:8000,:]
eval_labels1 = eval_labels[0:8000,:]

train_data2 = train_data[44000:55000,:]
train_labels2 = train_labels[44000:55000,:]
eval_data2 = eval_data[8000:10000,:]
eval_labels2 = eval_labels[8000:10000,:]
print("20% data:", train_data2.shape, train_labels2.shape, eval_data2.shape, eval_labels2.shape)

# global num
num = -1
def get_train_data():
    global num
    num += 200
    x = train_data1[num%44000:(num+200)%44000,:]
    y = train_labels1[num%44000:(num+200)%44000,:]
    # global num += 1
    return x, y, num

# for i in range(0,2):
#     images, labels, number= get_train_data()
#     # train_step.run(feed_dict={x: images, y_: labels})
#     print(images, type(images), images.shape)
#     print(labels, type(labels), labels.shape)
#     print(number)

x_image = tf.reshape(x, [-1, 28, 28, 1])

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

# '''
# 定义第一卷积层和第一池化层
w_conv1 = weight_variables([5, 5, 1, 6])  # 注意这里卷积核的个数为6个 filter:[height,weight,in_channels,out_channels]
b_conv1 = bias_variables([6])
# 第一层卷积输出
conv1 = conv2d(x_image, w_conv1) #x_image:[-1,28,28,1] w_conv1: filter
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
w_fc1 = weight_variables([7*7*120, 80])
b_fc1 = bias_variables([80])
# 把即将进入全连接层的输入h_conv3重塑为一维向量
h_conv3_flat = tf.reshape(h_conv3, [-1, 7*7*120])
# 第六层全连接层的输出
h_fc1 = tf.nn.sigmoid(tf.add(tf.matmul(h_conv3_flat, w_fc1), b_fc1))

# 最后一层全连接层，使用softmax进行分类
w_fc2 = weight_variables([80, 10])
b_fc2 = bias_variables([10])
y_model = tf.nn.softmax(tf.add(tf.matmul(h_fc1, w_fc2), b_fc2))

# 损失函数，采用交叉熵
loss = -tf.reduce_sum(y_ * tf.log(y_model))
# 训练
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss) #training rate
# 准确率
correct_prediction = tf.equal(tf.argmax(y_model, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
train_c = []  # 用来承载训练准确率，以便于绘制准确率曲线
test_c = []  # 用来承载测试准确率，以便于绘制测试准确率曲线

# 启动训练过程
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()
start_time = time.time()

for i in range(2000):
    # 获取训练数据
    # 需要一个function代替next_batch
    # batch = mnist.train.next_batch(200) #每次提取200张照片训练 循环2000次
    images, labels, number = get_train_data()
    # 训练数据
    # train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    # print(batch[0], type(batch[0]), batch[0].shape)
    # print(images, type(images), images.shape)
    train_step.run(feed_dict={x: np.array(images), y_: np.array(labels)})
    # 每迭代100个batch，对当前训练数据进行测试，输出训练acc和测试acc
    if i % 2 == 0:
        # train_acc = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
        train_acc = accuracy.eval(feed_dict={x: images, y_: labels})
        # test_acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        test_acc = accuracy.eval(feed_dict={x: eval_data1, y_: eval_labels1})
        train_c.append(train_acc)
        test_c.append(test_acc)
        print("step %d: \ntraining accuracy %g, testing accuracy %g" % (i, train_acc, test_acc))
        # 计算间隔时间
        end_time = time.time()
        print("time: ", (end_time - start_time))
        start_time = end_time
    # save model
    if i % 100 == 0:
        print("save model index:", i)
        saver.save(sess, "/Users/apple/Desktop/foolbox/foolbox/MNIST_model/LeNet_model.ckpt", write_meta_graph=True)
        # path = saver.save(sess, "/Users/apple/Desktop/foolbox/foolbox/MNIS_data/model.ckpt", global_step=i)

# Test trained model
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(sess.run(accuracy, feed_dict={x: mnist.test.images,
#                                     y_: mnist.test.labels}))
print("test accuracy %g"%accuracy.eval(feed_dict={x: eval_data1, y_: eval_labels1}))

ckpt = tf.train.get_checkpoint_state("/Users/apple/Desktop/foolbox/foolbox/MNIST_model/")
saver.restore(sess, ckpt.all_model_checkpoint_paths[0])
print(ckpt)
print("model's test accuracy %g"%accuracy.eval(feed_dict={x: eval_data1, y_: eval_labels1}))
# print(sess.run(accuracy, feed_dict={x: mnist.test.images,
#                                     y_: mnist.test.labels}))

sess.close()
plt.plot(train_c, label="train accuracy")
plt.plot(test_c, label="test accuracy")
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig("tensorflow_application/accu.png", dpi=200)
# '''



