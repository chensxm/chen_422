# 异或神经网络BP
import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777)  #设置随机种子
learning_rate = 0.1
# 定义数据集
x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
y_data = [[0],
          [1],
          [1],
          [0]]
#定义占位符
X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])
#模型和前向传播
W1 = tf.Variable(tf.random_normal([2, 3]), name='weight1')
b1 = tf.Variable(tf.random_normal([3]), name='bias1')
a1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([3, 1]), name='weight2')
b2 = tf.Variable(tf.random_normal([1]), name='bias2')
a2 = tf.sigmoid(tf.matmul(a1, W2) + b2)
# 代价或损失函数
cost = -tf.reduce_mean(Y * tf.log(a2) + (1 - Y) * tf.log(1 - a2))
cost_history = [] # 损失值列表
# #BP反向传播
# #第2层
# dz2 = a2 - Y
# dW2 = tf.matmul(tf.transpose(a1), dz2) / tf.cast(tf.shape(a1)[0], dtype=tf.float32)
# db2 = tf.reduce_mean(dz2)
# #第1层
# da1 = tf.matmul(dz2, tf.transpose(W2))
# dz1 = da1 * a1 * (1 - a1)
# dW1 = tf.matmul(tf.transpose(X), dz1) / tf.cast(tf.shape(X)[0], dtype=tf.float32)
# db1 = tf.reduce_mean(dz1, axis=0)
# # 参数更新
# update = [
#   tf.assign(W2, W2 - learning_rate * dW2),
#   tf.assign(b2, b2 - learning_rate * db2),
#   tf.assign(W1, W1 - learning_rate * dW1),
#   tf.assign(b1, b1 - learning_rate * db1)
# ]
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
# 准确率计算
predicted = tf.cast(a2 > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
# 创建会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #全局变量初始化
    # 迭代训练
    for step in range(30001):
        _, cost_val, acc_val = sess.run([train, cost, accuracy], feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:# 显示损失值收敛情况
            print(step, "Cost: ", cost_val, acc_val)
            cost_history.append(cost_val)
    # 画学习曲线
    plt.plot(cost_history[1: len(cost_history)])
    plt.show()


