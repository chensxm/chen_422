# -*- coding: utf-8 -*-
# @Time    : 2024/2/22 11:21
# @Author  : xk
# @File    :日考3.py
# @Software: PyCharm
# 在tensorflow框架下编写python代码
import tensorflow as tf
from warnings import filterwarnings
filterwarnings('ignore')
# （一）定义变量a，值为[[2,3,4],[3,4,5]] （8分）
a = tf.Variable([[2, 3, 4], [3, 4, 5]])
# （二）定义变量b，值为[[2,3],[3,4],[4,5]] （8分）
b = tf.Variable([[2, 3], [3, 4], [4, 5]])
# （三）定义a数字乘以2的操作（8分）
a_m_2 = tf.multiply(a, 2)
# （四）定义a矩阵乘以b的操作（8分）
a_m_b = tf.matmul(a, b)
mub = tf.Variable(1, dtype=tf.int32)
# （五）创建Session对象（8分）
with tf.Session() as sess:
    # （六）执行全局变量初始化（8分）
    sess.run(tf.global_variables_initializer())
    # （七）输出变量a的值（8分）
    print('变量a\n', sess.run(a))
    # （八）输出变量b的值（8分）
    print('变量b\n', sess.run(b))
    # （九）执行a数字乘以2的操作，输出结果（8分）
    print('a数字乘以2\n', sess.run(a_m_2))
    # （十）执行a矩阵乘以b的操作，输出结果（8分）
    print('a矩阵乘以b的\n', sess.run(a_m_b))
    # （十一）完成数字10的阶乘运算，输出每个步骤的结果。（每个步骤5分，至少4个步骤，共20分）
    for i in range(1, 11):
        print(sess.run(tf.assign(mub, tf.multiply(mub, i))))




