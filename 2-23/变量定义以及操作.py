# -*- coding: utf-8 -*-
# Time : 2024/2/23 10:27
# Author : chen
# Software: PyCharm
# File : 变量定义以及操作.py
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf

#创建一个变量
x = tf.Variable([1,2])

#创建一个常量
y = tf.constant([3,3])

#减法操作
sub = tf.subtract(x,y)

print(sub)

#加法
add = tf.add(x,y)

#全局变量初始化 init   原因：---->  赋值
init = tf.global_variables_initializer()
with tf.Session() as sess:
    glob = sess.run(init)
    rss_add,rss_sub = sess.run([add,sub])

print(rss_add)
print(rss_sub)







