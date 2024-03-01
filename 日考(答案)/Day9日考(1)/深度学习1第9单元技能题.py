import tensorflow as tf
import numpy as np
tf.set_random_seed(777) #设置随机种子
#定义数据集
xy = np.loadtxt('data-04-zoo.csv', delimiter=',')
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
nb_classes = 7  # 0 ~ 6
#定义占位符
X = tf.placeholder("float", shape=[None, 16])
Y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 6
Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot
'''
[
    [[1,0,0,0,0,0,0]]
    [[0,1,0,0,0,0,0]]
]
'''
# print(Y_one_hot.shape)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
'''
[
    [1,0,0,0,0,0,0]
    [0,1,0,0,0,0,0]
]
'''
# print(Y_one_hot.shape)
#权重和偏置
W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')
#预测模型
logits = tf.matmul(X, W) + b
hypothesis = logits #预测模型之一，只用于softmax_cross_entropy_with_logits
# hypothesis = tf.nn.softmax(logits)  #预测模型之二，都可用
#代价或损失函数
# cost = tf.reduce_mean(-tf.reduce_sum(Y_one_hot * tf.log(hypothesis), axis=1))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot))
#梯度下降优化器
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
#准确率计算
prediction = tf.argmax(logits, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#创建会话
sess = tf.Session()
sess.run(tf.global_variables_initializer()) #全局变量初始化
#迭代训练
for step in range(1501):
    cost_val, _, acc = sess.run([cost, train, accuracy], feed_dict={X: x_data, Y: y_data})
    if step % 100 == 0:# 显示损失值收敛情况
        print(step, cost_val, acc)
#准确率
h, c, a = sess.run([hypothesis, prediction, accuracy], feed_dict={X: x_data, Y: y_data})
print("\nHypothesis: ", h[:3], "\nCorrect (Y): ", c[:3], "\nAccuracy: ", a)
# 测试
h1, p1 = sess.run([hypothesis, prediction],
    feed_dict={X: [[1,0,0,1,0,0,0,1,1,1,0,0,8,1,0,1]]})
print(h1, p1)
# while True:
#     str = input()
#     try:
#         if str == 'q':
#             break
#         test = list(map(float,str.split(',')))
#         h1, p1 = sess.run([hypothesis, prediction], feed_dict={X: [test]})
#         print(h1, p1)
#     except:
#         continue
