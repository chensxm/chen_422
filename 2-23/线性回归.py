# -*- coding: utf-8 -*-
# Time : 2024/2/23 11:12
# Author : chen
# Software: PyCharm
# File : 线性回归.py
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
#创建x_data--->随机创建100个数
x_data = np.random.rand(100)
#创建y_data
y_data = 0.1*x_data+0.2
#构造线性模型
# k
k = tf.Variable(0.)
# b
b = tf.Variable(0.)
# y
y = k * x_data + b
#构建二次代价函数
cost_loss = tf.reduce_mean(tf.square(y - y_data))#h--->预测# y-->真实值
#构建优化器  优化的是代价值
G = tf.train.GradientDescentOptimizer(0.2).minimize(cost_loss)
#初始化全局变量
init = tf.global_variables_initializer()
#训练模型
with tf.Session() as sess:
    sess.run(init)
    #梯度下降的核心是什么----->基于初始值一直循环
    for step in range(2000):
        sess.run(G)
        if step % 20 == 0:
            k_2201,b_2201,loss_2201 = sess.run([k,b,cost_loss])
            #格式化
            print('k={},b={},cost_loss={}'.format(k_2201,b_2201,loss_2201))


































