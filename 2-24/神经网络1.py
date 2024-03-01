# -*- coding: utf-8 -*-
# Time : 2024/2/24 9:05
# Author : chen
# Software: PyCharm
# File : 神经网络1.py
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 设置随机数种子
tf.set_random_seed(123)

# 读取MNIST数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)#one_hot以数组的形式出现

# 设置占位符
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 设置第一层参数
w1 = tf.Variable(tf.random_normal([784, 256]))
b1 = tf.Variable(tf.zeros([256]))

# 设置第二层参数
w2 = tf.Variable(tf.random_normal([256, 10]))
b2 = tf.Variable(tf.zeros([10]))


# 定义预测函数
def h(x):
    layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    layer2 = tf.matmul(layer1, w2) + b2
    return layer2


# 定义代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h(x), labels=y))

# 设置优化器
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# 定义准确率
correct_prediction = tf.equal(tf.argmax(h(x), 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 建立会话并初始化变量
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 设置大循环10次
for epoch in range(10):
    # 设置批次大小为100   
    batch_size = 100#批次的大小，神经网络经历一次迭代的需要计算的次数，神经网络经历一个迭代需要计算batchsize次
    # 计算总批次数
    total_batch = mnist.train.num_examples // batch_size

    # 循环读取批次数据
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, _loss = sess.run([optimizer, loss], feed_dict={x: batch_x, y: batch_y})

        # 输出每批次损失值
        print("Epoch:", '%02d' % (epoch + 1), "Batch:", '%03d' % (i + 1), "Loss:", _loss)

# 计算测试集精度
print("Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
