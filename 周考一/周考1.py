# -*- coding: utf-8 -*-
# Time : 2024/2/26 10:50
# Author : chen
# Software: PyCharm
# File : 周考1.py

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression
# 1.以上给予的的背景小知识的了解，请使用TensorFlow完成以下相关的题目要求。
# (1)以下为一个判断逻异或的的数据，按照要求去做逻辑回归运算；
# ①正确加载下图给予的亦或初始化数据（7分）
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
# ②合理的运用tf.placeholder进行定义（7分）  placeholder:占位  作用：
X = tf.placeholder(tf.float32,shape=[None,2])
y = tf.placeholder(tf.float32,shape=[None,1])
# ③合理的根据以上数据进行偏执和权重的设置，注意维度问题。（7分）
# y = kx + b
w = tf.Variable(tf.random_normal([2,1]),name='weight')#
b = tf.Variable(tf.random_normal([1]),name='bigs')#
# ④调用tf.sigmoid模块完成预测模型（7分）
# y = wx + b
model = tf.matmul(X,w) + b
h = tf.sigmoid(model)#预测值
predict = tf.cast(h>0.5,dtype=tf.int32)
# ⑤用底层写出损失函数，注意是交叉熵分类。（7分）
# loss = tf.reduce_mean(
#     tf.nn.sigmoid_cross_entropy_with_logits
#     (logits=sigmoid_model,labels=y))
loss = -tf.reduce_mean(y * tf.log(h) + (1-y) * tf.log(1-h))
# ⑥定义梯度下降（可以选择优化器种类也行）（7分）  lr
G = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
# ⑦创建会话，进行运算计算图分析。（7分）
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
# ⑧进行迭代运算，要求1000次
    for step in range(1000):
        cost_function,_ = sess.run([loss,G],feed_dict={X:x_data,y:y_data})
# ⑨合理的步数（可以是40步）给出损失值结果。（7分）
        if step % 40 == 0:
            print('step={}','loss={}'.format(step,loss))
# ⑩最后进入验证预测功能，要求加入正确的的注释（7分）
    print(sess.run(predict,feed_dict={X:x_data}))
# 2.有关Tensorflow的常见的运算；
# ①定义二维的数组[[1.,3],[7.,5.],[9.,11.]]（6分）
# ②将二维数组转换为一维的向量（6分）
# ③对①按照要求算出每行和整个数组的平均值（6分）
# ④对①及求第二行最大的值，第一列最小的值。（6分）
# 对数组①进行数字翻倍，且输出结果（6分







