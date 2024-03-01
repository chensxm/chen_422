import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

# 加载Iris数据集
data = load_iris()
X = data.data
y = data.target

# 特征缩放
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 定义输入placeholder，对label进行one-hot处理
X_placeholder = tf.placeholder(tf.float32, shape=[None, 4])
y_placeholder = tf.placeholder(tf.int32, shape=[None])
y_onehot = tf.one_hot(y_placeholder, depth=3)

# 定义网络结构
hidden1 = tf.layers.dense(X_placeholder, units=50, activation=tf.nn.relu)
hidden2 = tf.layers.dense(hidden1, units=50, activation=tf.nn.relu)
output = tf.layers.dense(hidden2, units=3)

# 定义损失函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=output))

# 计算精度
correct_predictions = tf.equal(tf.argmax(output, 1), tf.argmax(y_onehot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# 定义优化器
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)

# 训练模型
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        _, train_loss, train_accuracy = sess.run([train_op, loss, accuracy],
                                                 feed_dict={X_placeholder: X_train, y_placeholder: y_train})
        if i % 100 == 0:
            print(f"Step {i}, Loss: {train_loss}, Accuracy: {train_accuracy}")
