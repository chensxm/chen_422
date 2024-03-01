# -*- coding: utf-8 -*-
# Time : 2024/2/22 15:52
# Author : chen
# Software: PyCharm
# File : tensorflow复习.py
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
# a + b = c
# a = 1
# b = 2
# def demo():
#     a = 1
#     b = 2
#     c = a+b
#     print(c)
# demo()
#利用tensorflow构建一个加法
a = tf.constant(2,dtype=tf.int32)
b = tf.constant(3)
c = tf.add(a,b)
print(c)
with tf.Session() as sess:
    print(sess.run(c))
# sess.close()
#with
# 上下文管理























