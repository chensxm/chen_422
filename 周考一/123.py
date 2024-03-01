# -*- coding: utf-8 -*-
# Time : 2024/2/26 11:43
# Author : chen
# Software: PyCharm
# File : 123.py
import warnings

warnings.filterwarnings('ignore')
# 1.以上给予的的背景小知识的了解，请使用TensorFlow完成以下相关的题目要求。
# (1)以下为一个判断逻异或的的数据，按照要求去做逻辑回归运算；
# ①正确加载下图给予的亦或初始化数据（7分）
import tensorflow as tf
import numpy as np

tf.set_random_seed(888)
x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]
print(np.array(x_data).shape)
print(np.array(y_data).shape)
# #
# #
# # ②合理的运用tf.placeholder进行定义（7分）
X = tf.placeholder(dtype=tf.float32, shape=[None, 2])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
# ③合理的根据以上数据进行偏执和权重的设置，注意维度问题。（7分）
w = tf.Variable(tf.random_normal(shape=[2, 1]))
b = tf.Variable(tf.random_normal(shape=[1]))
# # ④调用tf.sigmoid模块完成预测模型（7分）
h = tf.sigmoid(tf.matmul(X, w) + b)
predict = tf.cast(h > 0.5, dtype=tf.int32)
# # ⑤用底层写出损失函数，注意是交叉熵分类。（7分）
loss = -tf.reduce_mean(Y * tf.log(h) + (1 - Y) * tf.log(1 - h))
# # ⑥定义梯度下降（可以选择优化器种类也行）（7分）
op = tf.train.AdamOptimizer(0.01).minimize(loss)
# # ⑦创建会话，进行运算计算图分析。（7分）
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # ⑧进行迭代运算，要求1000次
    for i in range(1000):
        loss_, op_ = sess.run([loss, op], feed_dict={X: x_data, Y: y_data})
        # ⑨合理的步数（可以是40步）给出损失值结果。（7分）
        if i % 40 == 0:
            print(i, loss_)
    # ⑩最后进入验证预测功能，要求加入正确的的注释（7分）
    print(sess.run(predict, feed_dict={X: x_data}))
