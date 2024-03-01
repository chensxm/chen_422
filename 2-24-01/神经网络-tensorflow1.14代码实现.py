# -*- coding: utf-8 -*-
# Time : 2024/2/24 11:12
# Author : chen
# Software: PyCharm
# File : 神经网络-tensorflow1.14代码实现.py
import warnings
import numpy as np
warnings.filterwarnings('ignore')
# 使用神经网络实现(数据集MNIST_data):
# (1)导入tf库(2分)
import tensorflow as tf#tensorflow框架
import tensorflow.examples.tutorials.mnist.input_data#读取数据
from tensorflow.examples.tutorials.mnist import input_data
# (2)设置随机数种子(2分)
tf.set_random_seed(888)#设置随机树的种子原因：每次运行的结果一致
# (3)调用函数input_data包中的read_data_sets读取手写数字识别data(2分)
mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)#one_hot独热成数组  ndarray()
#打印数据
# (4)设置x占位符(2分)
# (5)设置y占位符，并对其做tf.one_hot处理(2分)
x = tf.placeholder(tf.float32,shape=[None,784])
y = tf.placeholder(tf.float32,shape=[None,10])
# (6)设置第一层参数w1,设置参数b1(2分)
w1 = tf.Variable(tf.random_normal(shape=[784,256]))
b1 = tf.Variable(tf.zeros(256))
# (7)设置第二层参数w2,设置参数b2((2分)
w2 = tf.Variable(tf.random_normal(shape=[256,10]))
b2 = tf.Variable(tf.zeros(10))
# (8)定义预测函数h(2分)
def h(x):
    #定义两层
    layer1 = tf.nn.relu(tf.matmul(x,w1)+b1)#tf.nn.relu--->
    layer2 = tf.matmul(layer1,w2) + b2
    return layer2
    pass
# (9)定义代价函数loss(2分)
loss = tf.nn.softmax_cross_entropy_with_logits(logits=h(x),labels=y)
# (10)设置优华器(2分)--->梯度
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
# (11)定义准确率(2分)
# def accurary(a3,y):
#     eq = np.equal(a3,y)
#     eq_mean = np.mean(eq)
#     return eq_mean
# axis = 0  横向操作
# axis = 1  纵向操作
Accuracy = tf.equal(tf.argmax(h(x),1),tf.argmax(y,1))
# (12)建立session，并进行全局变量初始化(2分)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
acc = tf.reduce_mean(tf.cast(Accuracy,tf.float32))
# (13)设置大循环10次（epoch=10）(2分)

for epoth in range(10):
    # (14)设置批次大小为100（batch_size=100）(2分)
    batch_size = 100# 在tensorflow当中为什么要设置批次：原因是：
    # (15)计算总批次数batch(2分)
    # total_batch1 = len(x)//batch_size
    total_batch = mnist.train.num_examples//batch_size
    # (16)调用mnist数据集自带的next_batch循环读取批次数据(2分)
    for i in range(total_batch):
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        _,_loss = sess.run([optimizer,loss],feed_dict={x:batch_x,y:batch_y})
    # (17)输出每批次损失值(2分)
    #     print('epoch','%0.2d' % (epoth+1),"batch",'%03d' % (i+1),'loss',_loss)
    #     print(i)
# (18)计算测试集精度。(2分)
print('测试集精度',sess.run(acc,feed_dict={x:mnist.test.images,y:mnist.test.labels}))




