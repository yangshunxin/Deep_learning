''''
基本概念讲解：
    TensorFlow中和RNN相关的API主要位于两个package: tf.nn.rnn_cell(主要定义RNN的常见的几种细胞cell)、tf.nn(RNN相关的辅助操作)
    一、RNN中的细胞Cell
        基类：tf.nn.rnn_cell.RNNCell
        最基本的RNN的实现Cell：tf.nn.rnn_cell.BasicRNNCell
        简单的LSTM Cell实现：tf.nn.rnn_cell．BasicLSTMCell
        最常用的LSTM Cell实现：tf.nn.rnn_cell.LSTMCell
        GRU Cell实现：tf.nn.rnn_cell.GRUCell
        多层RNN结构网络的实现：tf.nn.rnn_cell.MultiRNNCell
'''
import tensorflow as tf
#    定义Cell
#        num_units: 给定各个神经层次中的神经元数目（状态维度和输出的数据维度和num_units一致）
#                    也就是说一个神经元，输出一个状态，那么输出的维度就跟状态？？？？？
#           Most basic RNN: output = new_state = act(W * input + U * state + B)
# cell = tf.nn.rnn_cell.BasicRNNCell(num_units=128)
# # print(cell.state_size)
# # print(cell.output_size)
# # # 4 表示的是每个时刻输入4个样本，64表示每个样本具有64维的特征
# inputs = tf.placeholder(tf.float32, shape=(4,64))
# # # 给定RNN的初始状态，4表示每个时刻输入的样本数目
# # #　s0 的状态是：（batch_size, state_size）
# s0 = cell.zero_state(4, tf.float32)
# print(s0.get_shape())
# # # 对于t=1时刻传入输入和state获取结果值
# output, s1 = cell.call(inputs, s0)
# print(s1.get_shape())
# print(output.get_shape())

# ## 定义lstm cell
# lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=122)
# ## 4表示的是每个时刻输入4个样本，64表示每个样本具有64维的特征
# inputs = tf.placeholder(tf.float32, shape=(4, 64))
# ## 给定RNN的初始状态，4表示每个时刻输入的样本数目
# ## s0的状态是：（batch_size, state_size）
# s0 = lstm_cell.zero_state(4, tf.float32)
# print(s0.h.get_shape())
# print(s0.c.get_shape())
# # 对于t=1时刻传入输入和state获取结果值(LSTM中存在两个传入到下一个时刻的隐状态，即：C和h，在API中，全部都存储于s1中)
# output, s1 = lstm_cell.call(inputs, s0) # ===> 等价于output, s1 = lstm_cell(inputs, s0)
# print(s1.h.get_shape())
# print(s1.c.get_shape())
# print(output.get_shape())

## 一次多布的执行
## 因为cell.call方法，需要每个时刻均调用一次，比较麻烦
# inputs：输入信息，一组序列（从t=0到t=T）,格式要求：[batch_size, time_steps, input_size],batch_size：每个时刻输入的样本数目，time_steps:序列长度（时间长度）， input_size:输入数据中单个样本的维度数量
## initial_state：初始状态，一般为0矩阵
# 返回：output: time_steps所有的输出，格式为: [batch_size, time_steps, output_size]
# 返回：state：最后一步的状态，格式为: [batch_size, state_size]
# output, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)

import  numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 数据加载（每个样本是784维）
mnist = input_data.read_data_sets('data/', one_hot=True)

# 构建一个会话
with tf.Session() as sess:
    lr = 0.001 # 学习率
    input_size = 28 # 每个时刻输入的数据维度大小
    timestep_size = 28 # 时刻数目，总共输入多少个时刻
    hidden_size = 128 # 细胞中一个神经网络的层次中的神经元的数目
    layer_num = 2 # RNN中的隐层的数目
    class_num = 10 # 最后输出的类别

    _X = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, class_num])
    # batch_size是一个int32类型的标量tensor的占位符，使用batch_size可以让我们在训练和测试的时候使用不同的数据量
    batch_size = tf.placeholder(tf.int32, [])
    # dropout的时候，保留率多少
    keep_prob = tf.placeholder(tf.float32, [])

    # 开始构建网络、
    # 1. 输入的数据格式转换
    # X格式：[batch_size, time_steps, input_size]
    X = tf.reshape(_X, shape=[-1, timestep_size, input_size])

    # 2. 定义cell  每个cell中有 128个units
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, reuse=tf.get_variable_scope().reuse)

    # 3. 单层RNN 网络应用
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32) # batch_size 表示输入数据的批次数量

    # time_major=False, 默认就是False， output的格式为：
    #               [batch_size, timestep_size, hidden_size], 获取最后一个时刻的输出值是：output_ = output[:,-1,:] 一般就是默认值
    outputs, state = tf.nn.dynamic_rnn(lstm_cell, inputs=X, initial_state=init_state)
    output = outputs[:, -1, :]
    # print(outputs.shape) # (?, 28, 128)

    # 将输出值(最后一个时刻对应的输出值构建加下来的全连接)
    w = tf.Variable(tf.truncated_normal([hidden_size, class_num], mean=0.0, stddev=0.1), dtype=tf.float32, name='out_w')
    b = tf.Variable(tf.constant(0.1, shape=[class_num]), dtype=tf.float32, name='out_b')
    y_pre = tf.nn.softmax(tf.matmul(output, w) + b)

    # 损失函数定义
    loss = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_pre), 1))
    train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    # 准确率
    cp = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(cp, 'float'))

    # 开始训练
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        _batch_size = 128
        batch = mnist.train.next_batch(_batch_size)
        # 训练模型
        sess.run(train, feed_dict={_X: batch[0], y: batch[1], keep_prob: 0.5, batch_size: _batch_size})
        # 隔一段时间计算一下准确率
        if (i + 1) % 200 == 0:
            train_acc = sess.run(accuracy,
                                 feed_dict={_X: batch[0], y: batch[1], keep_prob: 1.0, batch_size: _batch_size})
            print("批次:{}, 步骤:{}, 训练集准确率:{}".format(mnist.train.epochs_completed, (i + 1), train_acc))

    # 测试集准确率计算
    test_acc = sess.run(accuracy, feed_dict={_X: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0,
                                             batch_size: mnist.test.num_examples})
    print("测试集准确率:{}".format(test_acc))


