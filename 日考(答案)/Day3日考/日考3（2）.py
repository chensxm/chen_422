# -*- coding: utf-8 -*-
# @Time    : 2024/2/22 11:30
# @Author  : xk
# @File    :日考3（2）.py
# @Software: PyCharm
# 在tensorflow框架下编写python代码
import tensorflow as tf
from warnings import filterwarnings
filterwarnings('ignore')

# 1.定义变量a，值为[[1,3,5],[3,6,9]] （8分）
a = tf.Variable([[1, 3, 5], [3, 6, 9]])

# 2.定义变量b，值为[[2,7],[3,8],[2,6]] （8分）
b = tf.Variable([[2, 7], [3, 8], [2, 6]])

# 3.定义a数字乘以3的操作（8分）
a_m_3 = tf.multiply(a, 3)

# 4.定义a矩阵乘以矩阵b的操作（8分）
a_m_b = tf.matmul(a, b)
mul = tf.Variable(1, dtype=tf.int32)

# 5.创建Session对象（8分）
with tf.Session() as sess:

    # 6.执行全局变量初始化（8分）
    sess.run(tf.global_variables_initializer())

    # 7.输出变量a的值（8分）
    print('变量a\n', sess.run(a))

    # 8.输出变量b的值（8分）
    print('变量b\n', sess.run(b))

    # 9.执行a数字乘以3的操作，输出结果（8分）
    print('a数字乘以3\n', sess.run(a_m_3))

    # 10.执行a矩阵乘以矩阵b的操作，输出结果（8分）
    print('a矩阵乘以矩阵b\n', sess.run(a_m_b))

    # 11.完成数字10以内的奇数的阶乘运算，输出每个步骤的结果。（每个步骤5分，至少4个步骤，共20分）
    for i in range(1, 11):
        if i % 2 != 0:
            print(sess.run(tf.assign(mul, tf.multiply(mul, i))))
