# 5.	利用tensorflow框架，实现神经网络多分类问题：
# 导入tensorflow模块，设置随机种子
# 准备训练数据集x_data、y_data，从文件data-04-zoo.csv中读取
# 定义张量X和Y，float32类型，使用占位符函数
# 把Y转换为独热编码
# 定义张量W（weight）和b（bias）
# 定义hypothesis预测模型
# 定义代价函数（损失函数）
# 使用梯度下降优化器计算最小代价，查找最优解
# 创建会话（Session），全局变量初始化
# 开始迭代总共2001次
# 使用训练集的数据进行训练
# 每100次输出一次cost值
# 使用最后一个训练集的样本做为测试样本，进行分类测试，输出分类结果


import tensorflow as tf
import numpy as np

tf.set_random_seed(777)  # 设置随机种子
# 定义数据集
data = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=int)
x_data = data[:, :-1]
y_data = data[:, -1]
# 这两行代码的意思是将一个具有n个类别的标签数据y_data转换为一个one-hot编码的形式。
# 第一行定义了类别的数量为7个，第二行利用np.eye函数将原始的标签数据y_data转换为一个
# one-hot编码的矩阵。这种编码方式将每个类别用一个长度为n的向量表示，
# 其中对应的类别位置为1，其他位置为0。这种编码适用于神经网络等模型的分类任务。
nb_classes = 7
y_data = np.eye(nb_classes)[y_data]

# print(nb_classes)
# 0 ~ 6
# print(y_data)

# 定义占位符
X = tf.placeholder(tf.float32, shape=[None, 16])
Y = tf.placeholder(tf.int32, [None, nb_classes])  # 0 ~ 6
# Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot
# print(Y_one_hot.shape)
'''
[
    [[1,0,0,0,0,0,0]]
    [[0,1,0,0,0,0,0]]
]
'''
# print(Y_one_hot.shape)
# Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
# Y_one_hot = tf.squeeze(Y_one_hot) #压缩掉1的维度
'''
[
    [1,0,0,0,0,0,0]
    [0,1,0,0,0,0,0]
]
'''
# print(Y_one_hot.shape)
# 权重和偏置
W1 = tf.Variable(tf.random_normal([16, 256]), name='weight')
b1 = tf.Variable(tf.random_normal([256]), name='bias')
h1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([256, nb_classes]))
b2 = tf.Variable(tf.random_normal([nb_classes]))
# 预测模型
logits = tf.matmul(h1, W2) + b2
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
# 梯度下降优化器
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
# 准确率计算
y_prediction = tf.argmax(logits, 1)
y_true = tf.argmax(Y, 1)
correct_prediction = tf.equal(y_prediction, y_true)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 创建会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 全局变量初始化
    # 迭代训练
    for step in range(1501):
        cost_val, _, acc = sess.run([cost, train, accuracy], feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:  # 显示损失值收敛情况
            print(step, cost_val, acc)
    # 准确率
    h, c, a = sess.run([logits, y_prediction, accuracy], feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h[:3], "\nCorrect (Y): ", c[:3], "\nAccuracy: ", a)
    # 测试
    _, p1 = sess.run([logits, y_prediction], feed_dict={X: [x_data[-1]]})
    print(p1)
