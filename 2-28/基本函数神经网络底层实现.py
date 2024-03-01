# -*- coding: utf-8 -*-
# Time : 2024/2/28 10:50
# Author : chen
# Software: PyCharm
# File : 基本函数神经网络底层实现.py
import numpy as np
import warnings
warnings.filterwarnings('ignore')
def H(x,theta):
    return np.dot(x,theta)
def sigmoid(h):
    s = 1.0/(1.0+np.exp(-h))
    return s
def FP(x,theta1,theta2):
    a1 = x
    h2 = H(a1,theta1)
    a2 = sigmoid(h2)
    h3 = H(a2,theta2)
    a3 = sigmoid(h3)
    return a1,a2,a3

#反向传播
def BP(a1,a2,a3,y,theta2):
    m = len(y)
    #输出层的误差
    dele3 = a3 - y
    #隐藏层的误差
    dele2 = np.dot(dele3,theta2.T) * a2 * (1-a2)
    #求隐藏层的梯度值
    deletheta2 = 1.0/ m * np.dot(a2.T,dele3)
    deletheta1 = 1.0/ m * np.dot(a1.T,dele2)
    return deletheta1,deletheta2
def loss(a3,y):
    return -np.mean(y * np.log(a3) + (1-y) * np.log(1-a3))
def train(x,y,lr,iter_max,hidden):
    xm,xn = x.shape
    ym,yn = y.shape
    theta1 = np.zeros((xn,hidden))
    theta2 = np.zeros((hidden,yn))
    for i in range(2000):
        #计算模型值
        a1, a2, a3 = FP(x,theta1,theta2)
        #计算代价值
        j = loss(a3,y)
        #计算梯度值
        deletheta1, deletheta2 = BP(a1, a2, a3, y, theta2)
        #参数更新
        theta1 = theta1 - deletheta1 * lr
        theta2 = theta2 - deletheta2 * lr


    pass


