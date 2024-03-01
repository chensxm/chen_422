# -*- coding: utf-8 -*-
# Time : 2024/2/28 10:50
# Author : chen
# Software: PyCharm
# File : tensorflow实现底层神经网络.py
import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
# 使用tensorflow框架，建立神经网络，包含1个隐藏层，使用底层代码实现BP反向传播，实现逻辑异或功能。
# （一）导入tensorflow模块，设置随机种子（8分）
import tensorflow as tf
tf.set_random_seed(888)
# （二）准备数据集（8分）
# x1,x2,y的数据是0或者1，如下表，x1和x2进行逻辑异或的结果是y
# x1	x2	Y
# 0	   0	0
# 0	   1	1
# 1	   0	1
# 1	   1	0
x_data = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]
y_data = [
    [0],
    [1],
    [1],
    [0],
]
# （三）初始化X，Y占位符（8分）
X = tf.placeholder(tf.float32,shape=[None,2])
Y = tf.placeholder(tf.float32,shape=[None,1])
# （四）初始化W1，b1张量（8分）正向传播
W1 = tf.Variable(tf.random_normal([2,3]))
b1 = tf.Variable(tf.random_normal([3]))
# （五）设置隐藏层layer1模型，使用sigmoid函数（8分）
a1 = tf.sigmoid(tf.matmul(X,W1) + b1)
# （六）初始化W2，b2张量（8分）
W2 = tf.Variable(tf.random_normal([3,1]))
b2 = tf.Variable(tf.random_normal([1]))
# （七）设置hypothesis预测模型（8分）
a2 = tf.sigmoid(tf.matmul(a1,W2) + b2)
# （八）设置代价函数（8分）
loss = -tf.reduce_mean(Y * tf.log(a2) + (1-Y) * tf.log(1-a2))
# （九）不能使用梯度下降优化器，自己编写底层代码实现BP反向传播
#反向传播一共两层
cost_function = []
delet2 = a2 - Y
delet2_w = tf.matmul(tf.transpose(a1),delet2)/tf.cast(tf.shape(a1)[0],dtype=tf.float32)
d_mean = tf.reduce_mean(delet2_w)
#第一层
da1 = tf.matmul(delet2,tf.transpose(W2))
dz2 = da1 * a1 * (1-a1)
delet1_w = tf.matmul(tf.transpose(X),dz2)/tf.cast(tf.shape(X)[0],dtype=tf.float32)
d_mean1 = tf.reduce_mean(dz2,axis=0)
#参数更新
learning_rate = 0.01
update = [
    tf.assign(W2,W2 - delet2_w * learning_rate),
    tf.assign(b2,b2 - d_mean * learning_rate),
    tf.assign(W1,W1 - delet1_w * learning_rate),
    tf.assign(b1,b1 - d_mean1 * learning_rate),
]
#准确率计算
predict_accuracy = tf.cast(a2>0.5,dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict_accuracy,Y),dtype=tf.float32))
# 7个公式对应7行代码，每行代码3分，（共21分）
# （十）创建会话，初始化全局变量（5分）
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
# （十一）迭代训练2000次，每200次输出一次cost（5分）
    for step in range(2000):
        _,cost_val,accuracy_score = \
            sess.run([update,loss,accuracy],feed_dict={X:x_data,Y:y_data})
        if step % 200 == 0:
            print(step,cost_val,accuracy_score)
        cost_function.append(cost_val)
# （十二）输出预测值、准确度（5分）
#绘制代价曲线
    plt.plot(cost_function)
    plt.show()




