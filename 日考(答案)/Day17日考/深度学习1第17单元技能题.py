import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn import dynamic_rnn
tf.set_random_seed(777)
# Teach hello: hihell -> ihello

# 建立字典
idx2char = ['h', 'i', 'e', 'l', 'o']
# 构造数据集
x_data = [0, 1, 0, 2, 3, 3]   # hihell
# x_data = [0, 0, 0, 0, 0, 0]   # hihell
# x_one_hot = [[[1, 0, 0, 0, 0],   # h 0
#               [0, 1, 0, 0, 0],   # i 1
#               [1, 0, 0, 0, 0],   # h 0
#               [0, 0, 1, 0, 0],   # e 2
#               [0, 0, 0, 1, 0],   # l 3
#               [0, 0, 0, 1, 0]]]  # l 3
x_one_hot = np.eye(5)[x_data].reshape(1, -1, 5)  #独热编码
print(x_one_hot.shape)
y_data = [[1, 0, 2, 3, 3, 4]]    # ihello  不用独热编码

#设置参数
batch_size = 1   # 批大小one sentence
sequence_length = 6  # 序列长度 |ihello| == 6
input_dim = 5  # 独热编码长度one-hot size

hidden_size = 8  # 隐藏层神经元数量output from the LSTM. 5 to directly predict one-hot
num_classes = 5 # 类别总数
learning_rate = 0.1  #学习率

#定义占位符
X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])  # X input_dim独热one-hot
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label

#建立模型
#定义LSTM单元
# cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True) #报警
cell = tf.contrib.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32) #设置初始状态0
outputs, _states = dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)
print('rnn输出', outputs.shape)  #(1,6,8)

# 全连接层
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
#print(X_for_fc.shape)  (6,8)
# fc_w = tf.get_variable("fc_w", [hidden_size, num_classes])
# fc_b = tf.get_variable("fc_b", [num_classes])
# outputs = tf.matmul(X_for_fc, fc_w) + fc_b
outputs = tf.contrib.layers.fully_connected(inputs=X_for_fc, num_outputs=num_classes, activation_fn=None)
# outputs = tf.contrib.layers.fully_connected(inputs=outputs, num_outputs=num_classes, activation_fn=None)
print('全连接输出', outputs.shape) #经过一层全连接 变为(?, 5) (6,5) [batch_size*sequence_length,num_classes]

# 改变维度准备计算序列损失reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
print(outputs.shape)  # (1,6,5)
# 计算序列损失
weights = tf.ones([batch_size, sequence_length]) # 所有的权重都是1 All weights are 1 (equal weights)
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
# loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=Y))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# 预测值
prediction = tf.argmax(outputs, axis=2) #最后的outputs是三维的 所以axis=2
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_data),tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        l, _, acc = sess.run([loss, train, accuracy], feed_dict={X: x_one_hot, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data, acc)

        #预测结果查字典后输出字符串 print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tPrediction str: ", result_str, ''.join(result_str))

        if acc >= 1.0:
            break
    # 用新数据测试
    t_data = [0, 2, 3, 3, 0, 1]  # hellhi
    result = sess.run(prediction, feed_dict={X: np.eye(5)[t_data].reshape(1, -1, 5)})
    print(result)
    result_str = [idx2char[c] for c in np.squeeze(result)]
    print("\tPrediction str: ", result_str, ''.join(result_str))

'''
0 loss: 1.5785435 prediction:  [[3 3 3 3 3 3]] true Y:  [[1, 0, 2, 3, 3, 4]] 0.5
	Prediction str:  ['l', 'l', 'l', 'l', 'l', 'l'] llllll
1 loss: 1.4213508 prediction:  [[3 3 3 3 3 3]] true Y:  [[1, 0, 2, 3, 3, 4]] 0.33333334
	Prediction str:  ['l', 'l', 'l', 'l', 'l', 'l'] llllll
2 loss: 1.278454 prediction:  [[3 3 3 3 3 3]] true Y:  [[1, 0, 2, 3, 3, 4]] 0.33333334
	Prediction str:  ['l', 'l', 'l', 'l', 'l', 'l'] llllll
3 loss: 1.1136963 prediction:  [[2 3 2 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]] 0.33333334
	Prediction str:  ['e', 'l', 'e', 'l', 'l', 'o'] elello
4 loss: 0.90376115 prediction:  [[2 0 2 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]] 0.6666667
	Prediction str:  ['e', 'h', 'e', 'l', 'l', 'o'] ehello
5 loss: 0.7068202 prediction:  [[2 0 2 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]] 0.8333333
	Prediction str:  ['e', 'h', 'e', 'l', 'l', 'o'] ehello
6 loss: 0.5666248 prediction:  [[1 0 2 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]] 0.8333333
	Prediction str:  ['i', 'h', 'e', 'l', 'l', 'o'] ihello
7 loss: 0.4362999 prediction:  [[1 0 2 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]] 1.0
	Prediction str:  ['i', 'h', 'e', 'l', 'l', 'o'] ihello
[[1 3 3 4 4 4]]
	Prediction str:  ['i', 'l', 'l', 'o', 'o', 'o'] illooo

'''
