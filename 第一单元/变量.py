# -*- coding: utf-8 -*-
# Time : 2024/2/21 14:36
# Author : chen
# Software: PyCharm
# File : 变量.py
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
#创建一个变量
x = tf.Variable([1,2])
#创建一个常量
h = tf.constant([3,3])
#增加一个减法
sub = tf.subtract(x,h)
#增加一个加法
add = tf.add(x,h)
#声明全局变量的初始化---->初始化所有点
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))


