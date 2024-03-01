# -*- coding: utf-8 -*-
# Time : 2024/2/28 14:44
# Author : chen
# Software: PyCharm
# File : 2-6.py
import warnings

warnings.filterwarnings('ignore')
# 7. 数据集中包含了7个类别共10000个样本的数据，每一行数据为一个样本，最后一列为类别标签，其余列为特征，请使用tensorflow框架实现神经网络算法：(z7数据集)
# 导入tensorflow框架

# 设置随机种子
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import *

tf.set_random_seed(seed=888)
# 使用numpy读取数据
data = np.loadtxt('z7.csv', delimiter=',')
# 使用切片技术提取特征数据
x_data = data[:, :-1]
# 对加载的数据及进行特征缩放
scaler = MinMaxScaler().fit(x_data)
# print((x_data[0]))
# 使用切片技术提取类别标签数据
y_data = data[:, -1]
# print(y_data)
nb_classes = 7
print(nb_classes)
# 设置特征数据X的占位符
X = tf.placeholder(tf.float32, [None, 16])
# 设置标签数据Y的占位符
Y = tf.placeholder(tf.int32, [None])
# 调用tf.one_hot处理(6)中数据
Y_onehot = tf.one_hot(Y, 7)
# 设计感知机模型，并初始化相应参数
W1 = tf.Variable(tf.random_normal([16, 256]))
b1 = tf.Variable(tf.random_normal([256]))
h1 = tf.sigmoid(tf.matmul(X, W1) + b1)
print(h1.shape)
W2 = tf.Variable(tf.random_normal([256, 7]))
b2 = tf.Variable(tf.random_normal([7]))
h2 = tf.matmul(h1, W2) + b2
print(h2.shape)
# 写出loss函数（交叉熵损失函数），计算损失。
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h2, labels=Y_onehot))
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

predict = tf.argmax(h2, axis=1)
actual = tf.argmax(Y_onehot, axis=1)
print(predict.shape)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, actual), dtype=tf.float32))

# 建立会话并调用初始化函数
# 使用(3)和(4)中的数据，循环训练模型3000次
# 每隔10次循环，打印损失值
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        # x_data = scaler.transform(x_data)
        # print(x_data)
        loss_, train_ = sess.run([loss, train], feed_dict={X: scaler.transform(x_data), Y: y_data})
        if i % 10 == 0:
            acc_ = sess.run(accuracy, feed_dict={X: scaler.transform(x_data), Y: y_data})
            print(i, loss_, acc_)
