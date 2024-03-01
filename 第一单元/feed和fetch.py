# -*- coding: utf-8 -*-
# Time : 2024/2/21 14:45
# Author : chen
# Software: PyCharm
# File : feed和fetch.py
import warnings

warnings.filterwarnings('ignore')


import tensorflow as tf
#Fetch
imput1 = tf.constant(3.0)
imput2 = tf.constant(2.0)
imput3 = tf.constant(5.0)

add = tf.add(imput2,imput3)
mul = tf.multiply(imput1,add)

with tf.Session() as sess:
    result = sess.run([mul,add])
    print(result)


#Feed
#创建占位符
input1 = tf.placeholder(tf.float32)#占位符
input2 = tf.placeholder(tf.float32)#占位符
output = tf.multiply(input1,input2)
with tf.Session() as sess:
    #feed的数据以字典存入
    print(sess.run(output,feed_dict={input1:[7.],input2:[2]}))