# -*- coding: utf-8 -*-
# Time : 2024/2/28 14:22
# Author : chen
# Software: PyCharm
# File : 2-3.py
import warnings

warnings.filterwarnings('ignore')
import tensorflow as tf
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 加载波士顿房价数据集
boston = datasets.load_boston()
X = boston.data
y = boston.target

# 数据预处理
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义模型
n_features = X_train.shape[1]
X_placeholder = tf.placeholder(tf.float32, shape=(None, n_features))
y_placeholder = tf.placeholder(tf.float32, shape=(None,))

W = tf.Variable(tf.zeros([n_features, 1]))
b = tf.Variable(tf.zeros([1]))

y_pred = tf.matmul(X_placeholder, W) + b
loss = tf.reduce_mean(tf.square(y_placeholder - y_pred))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# 模型训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        _, l, weights, bias = sess.run([optimizer, loss, W, b],
                                       feed_dict={X_placeholder: X_train, y_placeholder: y_train})

        if i % 100 == 0:
            print("Epoch: {}, Loss: {}".format(i, l))
            print("Weights: {}".format(weights))
            print("Bias: {}".format(bias))

    # 模型预测
    y_pred_test = sess.run(y_pred, feed_dict={X_placeholder: X_test})

