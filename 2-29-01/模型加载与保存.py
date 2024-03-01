# -*- coding: utf-8 -*-
# Time : 2024/2/29 10:51
# Author : chen
# Software: PyCharm
# File : 模型加载与保存.py
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 载入数据集MNIST_data
mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)#one_hot--->独热  独热的作用：数据之间的差异过大
# 每个批次100张照片
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size
# 定义两个placeholder
x = tf.placeholder(dtype=tf.float32,shape=[None,784])
y = tf.placeholder(dtype=tf.float32,shape=[None,10])
# 创建一个简单的神经网络，输入层784个神经元，输出层10个神经元
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,w) + b)
# 二次代价函数  1.交叉熵  2.均方差
loss = tf.reduce_mean(tf.square(y - prediction))
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#     labels=y,logits=prediction))
# 使用梯度下降法
# #1.随机梯度下降
# 2.小批次梯度下降
# 3.大批次梯度下降
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
# 初始化变量
inin = tf.global_variables_initializer()
# 结果存放在一个布尔型列表中
Accuracy_predict = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
#argmax返回一维最大值的下标
# 求准确率
Accuracy = tf.reduce_mean(tf.cast(Accuracy_predict,tf.float32))
#声明保存模型的方法
saver = tf.train.Saver()
with tf.Session() as sess:
    #将保存的模型重新加载训练
    saver.restore(sess,save_path='uu/my_uu.ckpt')
    print(sess.run(Accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))
    # sess.run(inin)
    # for eporch in range(10):
    #     for i in range(n_batch):
    #         batch_x,batch_y = mnist.train.next_batch(batch_size)#next_batch
    #         sess.run(train_op,feed_dict={x:batch_x,y:batch_y})
    #     #求精度
    #     acc = sess.run(Accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
    #     print(acc)
    # #保存模型
    # saver.save(sess,'uu/my_uu.ckpt')
#载入数据集
#设置每个批次的大小
#计算一共有多少个批次
#定义三个placeholder
#创建一个多层神经网络模型
#第一个隐藏层
#第二个隐藏层
#第三个隐藏层
#输出层
#定义交叉熵代价函数
#定义反向传播算法（使用梯度下降算法）
#结果存放在一个布尔型列表中(argmax函数返回一维张量中最大的值所在的位置)
#求准确率(tf.cast将布尔值转换为float型)
#创建会话
#初始化变量
#训练次数20
#测试数据计算出的准确率
#保证模型














# 保存模型
