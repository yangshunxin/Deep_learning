import tensorflow as tf

# 输入数据
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
mnist = read_data_sets('/data', one_hot=True)

# 定义网络的超参
learning_rate = 0.01 # 学习率
training_iters=20000 # 训练的数据量
batch_size = 128 # 每次训练多少数据
display_step = 10 # 每多少次显示一下当前状态

# 定义网络的结构参数
n_input = 784
n_classes = 10
dropout=0.75

# 设定数据占位符  ---定义输入数据的结构
x = tf.placeholder(tf.float32, [None, n_input]) # [None, n_input]: None:表示行不做限制， n_input: 表示输入的列的维度
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# 定义卷积操作 （Conv layer）
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x= tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

# 定义池化操作
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

# 局部归一化 -LRN
def norm(x, lsize=4):  # 后面还可以
    return tf.nn.lrn(x, lsize, bias=1.0, alpha=0.001/9.0, beta=0.75)

# 定义网络的权重和偏置参数
weights = {
    'wc1': tf.Variable(tf.random_normal([11, 11, 1, 96])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
    'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
    'wd1': tf.Variable(tf.random_normal([2 * 2 * 256, 4096])),
    'wd2': tf.Variable(tf.random_normal([4096, 4096])),
    'out': tf.Variable(tf.random_normal([4096, n_classes]))
}
biases={
    'bc1':tf.Variable(tf.random_normal([96])),
    'bc2':tf.Variable(tf.random_normal([256])),
    'bc3':tf.Variable(tf.random_normal([384])),
    'bc4':tf.Variable(tf.random_normal([384])),
    'bc5':tf.Variable(tf.random_normal([256])),
    'bd1':tf.Variable(tf.random_normal([4096])),
    'bd2':tf.Variable(tf.random_normal([4096])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}

# 定义Alexnet网络结构
def alex_net(x, wights, biases, dropout):
    # 输出的数据做reshape
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # 底层卷积计算（conv + relu + pool）
    # 卷积
    conv1 = conv2d(x, wights['wc1'], biases['bc1'])
    # 池化
    pool1 = maxpool2d(conv1, k=2)
    # 规范化，局部归一化
    # 局部归一化 时仿照生物学上的活跃的神经元对相邻神经元的抑制现象
    norm1 = norm(pool1)

    # 第二层卷积：
    conv2=conv2d(norm1,weights['wc2'],biases['bc2'])
    # 池化
    pool2=maxpool2d(conv2,k=2)
    norm2=norm(pool2)

    #第三层卷积
    conv3=conv2d(norm2,weights['wc3'],biases['bc3'])
    # 池化
    pool3=maxpool2d(conv3,k=2)
    norm3=norm(pool3)

    #第四层卷积
    conv4=conv2d(norm3,weights['wc4'],biases['bc4'])
    #第五层卷积
    conv5=conv2d(conv4,weights['wc5'],biases['bc5'])
    # 池化
    pool5=maxpool2d(conv5,k=2)
    norm5=norm(pool5)
    #可以再加上dropout

    #全连接1
    # 向量化
    fc1=tf.reshape(norm5,[-1,weights['wd1'].get_shape().as_list()[0]])
    fc1=tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
    fc1=tf.nn.relu(fc1)
    #dropout
    fc1=tf.nn.dropout(fc1,dropout)

    #全连接2
    ## 向量化
    fc2=tf.reshape(fc1,[-1,weights['wd2'].get_shape().as_list()[0]])
    fc2=tf.add(tf.matmul(fc2,weights['wd2']),biases['bd2'])
    fc2=tf.nn.relu(fc2)
    #dropout
    fc2=tf.nn.dropout(fc2,dropout)

    #out
    return tf.add(tf.matmul(fc2,weights['out']),biases['out'])

# 1.定义损失函数和优化器，并构建评估函数
# （1）构建模型
pred=alex_net(x,weights,biases,keep_prob)
# (2)损失函数和优化器
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# (3)评估模型
correct_pred = tf.equal(tf.arg_max(pred, 1), tf.arg_max(y, 1))
acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 训练
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step=1
    # 开始训练，直到达到training_iters
    while step*batch_size<training_iters:
        batch_x,batch_y=mnist.train.next_batch(batch_size)
        sess.run(optim,feed_dict={x:batch_x,y:batch_y,keep_prob:dropout})
        if step%display_step==0:
            # 显示一下当前的损失和正确率
            loss,acc_num=sess.run([cost,acc],feed_dict={x:batch_x,y:batch_y,keep_prob:dropout})
            print('Iter:%d,Loss:%f,Train Acc:%f'%(step*batch_size,loss,acc_num))
        step+=1
print('Optimization finished')


