import tensorflow as tf
tf.set_random_seed(777) #设置随机种子
#定义数据集
x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]
#定义占位符
X = tf.placeholder("float", shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 3])
#权重和偏置
W = tf.Variable(tf.random_normal([4, 3]), name='weight')
b = tf.Variable(tf.random_normal([3]), name='bias')
#预测模型
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
#代价或损失函数
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
#梯度下降优化器
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
#准确率计算
prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(y_data, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#创建会话
sess = tf.Session()
sess.run(tf.global_variables_initializer()) #全局变量初始化
#迭代训练
for step in range(5001):
    cost_val, _, acc = sess.run([cost, train, accuracy], feed_dict={X: x_data, Y: y_data})
    if step % 500 == 0:# 显示损失值收敛情况
        print(step, cost_val, acc)
#准确率
h, c, a = sess.run([hypothesis, prediction, accuracy], feed_dict={X: x_data, Y: y_data})
print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
#测试
h1, p1 = sess.run([hypothesis, prediction], feed_dict={X: [[1, 2, 3, 4]]})
print(h1, p1)
h2, p2 = sess.run([hypothesis, prediction], feed_dict={X: [[4,1,2,3], [3,2,4,5]]})
print(h2, '\n', p2)
while True:
    str = input()  # 比如：1,2,3,4
    try:
        if str == 'exit':
            break
        test = list(map(float,str.split(',')))
        h1, p1 = sess.run([hypothesis, prediction], feed_dict={X: [test]})
        print(h1, p1)
    except:
        continue
