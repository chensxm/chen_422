# -*- coding: utf-8 -*-
# Time : 2024/2/28 14:18
# Author : chen
# Software: PyCharm
# File : 2-2.py
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 导入MNIST数据
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
# 转换成tensorflow可以处理的数据
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

# 特征缩放处理
x_train = x_train / 255
x_test = x_test / 255

# 定义输入数据的placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])


# 定义网络模型结构
def neural_network(x):
    # 第一层
    W1 = tf.Variable(tf.random_normal([784, 256]))
    b1 = tf.Variable(tf.random_normal([256]))
    layer1 = tf.nn.relu(tf.matmul(x, W1) + b1)

    # 第二层
    W2 = tf.Variable(tf.random_normal([256, 128]))
    b2 = tf.Variable(tf.random_normal([128]))
    layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)

    # 输出层
    W_out = tf.Variable(tf.random_normal([128, 10]))
    b_out = tf.Variable(tf.random_normal([10]))
    output = tf.matmul(layer2, W_out) + b_out

    return output


# 前向预测函数
logits = neural_network(x)

# 定义损失函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

# 选择优化器和学习率
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss)

# 计算精度
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 会话功能
batch_size = 128
epochs = 2000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train_op, feed_dict={x: batch_x, y: batch_y})

        if epoch % 100 == 0:
            acc = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
            print("Epoch {}, Test Accuracy: {}".format(epoch, acc))
