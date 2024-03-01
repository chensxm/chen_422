# -*- coding: utf-8 -*-
# Time : 2024/2/28 15:52
# Author : chen
# Software: PyCharm
# File : 2-1.py
import warnings
warnings.filterwarnings('ignore')
# Iris数据集是sklearn自带的数据集，里面收录了3类鸢尾花，各50个样本，
# 每个样本由4类特征构成：花瓣长度和宽度，花萼长度和宽度。
# 1.现有题目如下：
# (1)题目分析：
# ①正确导入相关头文件的包。
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
# ②通过对以上关于Iris数据的分析，自动加载。
data = load_iris()
X = data.data
y = data.target
# ③对加载的数据及进行特征缩放
std = StandardScaler()
X = std.fit_transform(X)
train_x,test_x,train_y,test_y = train_test_split(X,y,test_size=0.2)

# ④定义tf.placeholder，对label进行one-hot处理。
x_placeholder = tf.placeholder(tf.float32,shape=[None,4])
y_placeholder = tf.placeholder(tf.int32,shape=[None])
y_onehot = tf.one_hot(y_placeholder,depth=3)
# ⑤加入两层隐藏层，层数自行设计。
# ⑥根据网络模型结构，设置每一层weight，bias。
#网络神经结构
# hidden1 = tf.Variable(tf.random_normal([4,50]))
# hidden2 = tf.Variable(tf.random_normal([50]))
hidden1 = tf.layers.dense(x_placeholder,units=50,activation=tf.nn.relu)
hidden2 = tf.layers.dense(hidden1,units=50,activation=tf.nn.relu)
output = tf.layers.dense(hidden2,units=3)
# ⑦写出loss函数（交叉熵损失函数），计算损失。
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels=y_onehot,logits=output))
# ⑧写出accuracy计算逻辑，计算精度
accuracy_predict = tf.equal(tf.argmax(output,1),tf.argmax(y_onehot,1))
accuracy = tf.reduce_mean(tf.cast(accuracy_predict,dtype=tf.float32))
# ⑨每100次打印步数，损失和精度；
#需要给优化器  优化的是什么：代价
optimizer = tf.train.AdamOptimizer().minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        _,train_loss,accuracy_pp = sess.run(
            [optimizer,loss,accuracy],feed_dict=
            {x_placeholder:train_x,y_placeholder:train_y})
        if i % 100 == 0:
            print(f'step:{i},loss:{train_loss},accuracy:{accuracy_pp}')

# ⑩对关键步骤给出注解
# ⑪代码逻辑过程清晰
