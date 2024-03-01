# -*- coding: utf-8 -*-
# Time : 2024/2/21 14:27
# Author : chen
# Software: PyCharm
# File : 计算图的创建与启动.py
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
# 创建一个常量
q1 = tf.constant([[3,3]])
q2 = tf.constant([[2],[3]])

# 创建一个矩阵乘法  将q1和q2相乘
stp = tf.matmul(q1,q2)
print(stp)

# 创建一个会话  启动默认图
sess = tf.Session()
#调用sess的run方法启动执行矩阵乘法
result = sess.run(stp)
print(result)
sess.close()


with tf.Session() as sess:
    # 调用sess的run方法启动执行矩阵乘法
    result = sess.run(stp)
    print(result)