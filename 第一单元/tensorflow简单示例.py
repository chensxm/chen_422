# -*- coding: utf-8 -*-
# Time : 2024/2/21 14:53
# Author : chen
# Software: PyCharm
# File : tensorflow简单示例.py
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']= '2'
import numpy as np
import tensorflow as tf
#使用numpy随机生成100个点
x_data = np.random.rand(100)
y_data = x_data * 0.1 + 0.2
#构造一个线性模型
b = tf.Variable(1.1)
k = tf.Variable(0.5)
y = k * x_data + b
#二次代价函数
loss = tf.reduce_mean(tf.square(y_data - y))#真-预
#定义一个梯度下降算法来进行训练的优化器
optimizer = tf.train.GradientDescentOptimizer(0.2)
#最小化代价函数
train = optimizer.minimize(loss)
#初始化变量
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        #每20次打印结果值
        if step % 20 == 0:
            print(step,sess.run([k,b]))






