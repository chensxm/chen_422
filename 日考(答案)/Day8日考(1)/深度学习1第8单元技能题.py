import tensorflow as tf
tf.set_random_seed(777) #设置随机种子
#定义数据集
x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]
#定义占位符
X = tf.placeholder("float", shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])
#权重和偏置
W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
#预测模型
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
#代价或损失函数
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
#梯度下降优化器
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
#准确率计算
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
#创建会话
sess = tf.Session()
sess.run(tf.global_variables_initializer()) #全局变量初始化
#迭代训练
for step in range(5001):
    cost_val, acc, _ = sess.run([cost, accuracy, train], feed_dict={X: x_data, Y: y_data})
    if step % 500 == 0:# 显示损失值收敛情况
        print(step, cost_val, acc)
#准确率
h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
#测试
h1, p1 = sess.run([hypothesis, predicted], feed_dict={X: [[1, 1]]})
print(h1, p1)
h2, p2 = sess.run([hypothesis, predicted], feed_dict={X: [[4,1], [3,100]]})
print(h2, '\n', p2)
while True:
    str = input()
    try:
        if str == 'q':
            break
        test = list(map(float,str.split(',')))
        h1, p1 = sess.run([hypothesis, predicted], feed_dict={X: [test]})
        print(h1, p1)
    except:
        continue
