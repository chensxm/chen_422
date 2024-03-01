# -*- coding: utf-8 -*-
# Time : 2024/2/28 14:43
# Author : chen
# Software: PyCharm
# File : 2-5.py
import warnings

warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np

tf.set_random_seed(888)
# (1)数据处理
# ①初始化训练数据[[0, 0], [0, 1], [1, 0], [1, 1]]，标签数据[[0], [1], [1], [0]]（6分）
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0], [1], [1], [0]]

print(np.array(x_data).shape)
print(np.array(y_data).shape)
# ②对标签数据进行维度变换操作，将其挤压为一位数组，利于one-hot进行处理（6分）
y_data = np.reshape(y_data, -1)
print(np.array(y_data).shape)

one_hoe_dim = len(set(y_data))
# (2)定义模型
# ①分别定义x，y两个占位符，对y进行one-hot变换（6分）
X = tf.placeholder(dtype=tf.float32, shape=(None, 2))
Y = tf.placeholder(dtype=tf.int32, shape=(None,))
Y_onehot = tf.one_hot(Y, one_hoe_dim)

# ②构建两层神经网络模型，隐藏层参数自定义（6分）
W1 = tf.Variable(tf.random_normal(shape=[2, 128]))
b1 = tf.Variable(tf.random_normal(shape=[128]))
h1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal(shape=[128, one_hoe_dim]))
b2 = tf.Variable(tf.random_normal(shape=[one_hoe_dim]))
h2 = tf.matmul(h1, W2) + b2

y_true = tf.argmax(Y_onehot, -1)
y_predict = tf.argmax(h2, -1)
acc = tf.reduce_mean(tf.cast(tf.equal(y_true, y_predict), tf.float32))
# ③计算损失函数，运用交叉熵（6分）
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h2, labels=Y_onehot))

# ④采用随机梯度优化算法，进行梯度下降，学习率设置为0.001-0.1之间的数值（6分）
op = tf.train.AdamOptimizer(0.01).minimize(loss)

# (3)模型训练
# ①生成会话，进行训练，最后需关闭会话（6分）
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # ②进行迭代训练，每迭代100次打印输出损失值和精度（6分）
    for i in range(3000):
        loss_, op_, acc_ = sess.run([loss, op, acc], feed_dict={X: x_data, Y: y_data})
        if i % 100 == 0:
            print(i, loss_, acc_)







