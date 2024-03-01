# -*- coding: utf-8 -*-
# Time : 2024/2/23 9:58
# Author : chen
# Software: PyCharm
# File : 计算数据流程图.py
from sklearn.datasets import load_breast_cancer#乳腺癌数据集
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
#创建两个常量 c1 c2
#变量  常量
c1 = tf.constant([[1,2]])#1*2的向量
c2 = tf.constant([[3],[4]])#2*1的向量
#实现乘法操作
matmul1 = tf.matmul(c1,c2)#dot()点乘
matmul2 = tf.matmul(c2,c1)#dot()点乘
with tf.Session() as sess:
    w1 = sess.run(matmul1)
    w2 = sess.run(matmul2)
    print(w1)
    print(w2)



