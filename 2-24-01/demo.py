# -*- coding: utf-8 -*-
# Time : 2024/2/24 11:49
# Author : chen
# Software: PyCharm
# File : demo.py
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
a = tf.constant([-1.0,2.0,3.6])
with tf.Session() as sess:
    b = tf.nn.relu(a)
    print(sess.run(b))

