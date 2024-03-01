# -*- coding: utf-8 -*-
# Time : 2024/2/27 11:56
# Author : chen
# Software: PyCharm
# File : 练习五.py
import warnings
warnings.filterwarnings('ignore')
# 1.按如下要求计算tensorflow运算
import tensorflow as tf
# (1)定义变量a，用正态分布随机初始化二维矩阵，二行三列 （6分）
a = tf.Variable(tf.random_normal([2,3]))
# (2)定义变量b，用正态分布随机初始化二维矩阵，三行二列（6分）
b = tf.Variable(tf.random_normal([3,2]))
# (3)定义a数字乘以3的操作（6分）
mul = a * 3
# (4)定义a矩阵乘以矩阵b的操作（8分）
a_b = tf.matmul(a,b)
# (5)创建Session对象（8分）
with tf.Session() as sess:
# (6)执行全局变量初始化（8分）
    sess.run(tf.global_variables_initializer())
# (7)执行a数字乘以3的操作，输出结果（8分）
    print(sess.run(mul))
# (8)执行a矩阵乘以矩阵b的操作，输出结果（8分）
    print(sess.run(a_b))
# (9)完成数字10以内的奇数的阶乘运算，输出每个步骤的结果。（6分）
# (10) 第(9)题中的阶乘运算要求使用赋值语句(6分)
# res = 1
# uu = 10
# for i in range(1,10):
#     if i % 2 == 1:
#         res *= i
# print(res,)

