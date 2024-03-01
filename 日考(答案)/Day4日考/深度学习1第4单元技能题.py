import tensorflow as tf
tf.set_random_seed(777) #设置随机种子
#定义数据集
x_data = [1, 2, 3]
y_data = [1.1, 2.11, 3.09]
#定义占位符
X = tf.placeholder("float", shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])
#权重和偏置
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
#预测模型
hypothesis = X * W + b
#代价或损失函数
cost = tf.reduce_mean(tf.square(hypothesis - Y))
#梯度下降优化器
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
#创建会话
sess = tf.Session()
sess.run(tf.global_variables_initializer()) #全局变量初始化
#迭代训练
for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X: x_data, Y: y_data})
    if step % 100 == 0:# 显示损失值收敛情况
        print(step, cost_val, W_val, b_val)
#验证
print(sess.run(hypothesis, feed_dict={X: [5]}))
print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))
sess.close()