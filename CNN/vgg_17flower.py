# -- encoding:utf-8 --
"""
17中花数据分类，是VGG网络初赛时候的数据集，现在网上没有下载；现在唯一一份数据集在tflearn这个框架中默认自带
tflearn这个框架起始是在tensorflow基础上的一个封装，API比较简单(如果代码功底比较好，建议用tensorflow)
tflearn安装：pip install tflearn
Create by ibf on 2018/5/20
"""

from tflearn.datasets import oxflower17  # pip install tflearn
import tensorflow as tf

# 读取数据
X, Y = oxflower17.load_data(dirname="17flowers", one_hot=True)
print(X.shape)  # sample_number,224,224,3
print(Y.shape)  # sample_number,17

# 相关的参数、超参数的设置
# 学习率，一般学习率设置的比较小
learn_rate = 0.1
# 每次迭代的训练样本数量
batch_size = 32
# 训练迭代次数(每个迭代次数中必须训练完一次所有的数据集)
train_epoch = 10000
# 样本数量
total_sample_number = X.shape[0]

# 模型构建
# 1. 设置数据输入的占位符
x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='x')
y = tf.placeholder(tf.float32, shape=[None, 17], name='y')


def get_variable(name, shape=None, dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0, stddev=0.1)):
    """
    返回一个对应的变量
    :param name:
    :param shape:
    :param dtype:
    :param initializer:
    :return:
    """
    return tf.get_variable(name, shape, dtype, initializer)


# 网络的构建
def vgg_network(x, y):
    net1_kernel_size = 32
    net3_kernel_size = 64
    net5_kernal_size_1 = 128
    net5_kernal_size_2 = 128
    net7_kernal_size_1 = 256
    net7_kernal_size_2 = 256
    net9_kernal_size_1 = 256
    net9_kernal_size_2 = 256
    net11_unit_size = 1000
    net12_unit_size = 1000
    net13_unit_size = 17

    # cov3-64 lrn
    with tf.variable_scope('net1'):
        net = tf.nn.conv2d(x, filter=get_variable('w', [3, 3, 3, net1_kernel_size]), strides=[1, 1, 1, 1],
                           padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b', [net1_kernel_size]))
        net = tf.nn.relu(net)
        # lrn(input, depth_radius=5, bias=1, alpha=1, beta=0.5, name=None)
        # 做一个局部响应归一化，是对卷积核的输出值做归一化
        # depth_radius ==> 对应ppt公式上的n，bias => 对应ppt公式上的k, alpha => 对应ppt公式上的α, beta=>对应ppt公式上的β
        net = tf.nn.lrn(net)
    # maxpool
    with tf.variable_scope('net2'):
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # conv3-128
    with tf.variable_scope('net3'):
        net = tf.nn.conv2d(net, filter=get_variable('w', [3, 3, net1_kernel_size, net3_kernel_size]),
                           strides=[1, 1, 1, 1],
                           padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b', [net3_kernel_size]))
        net = tf.nn.relu(net)
    # maxpool
    with tf.variable_scope('net4'):
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # conv3-256 conv3-256
    with tf.variable_scope('net5'):
        net = tf.nn.conv2d(net, filter=get_variable('w1', [3, 3, net3_kernel_size, net5_kernal_size_1]),
                           strides=[1, 1, 1, 1],
                           padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b1', [net5_kernal_size_1]))
        net = tf.nn.relu(net)

        net = tf.nn.conv2d(net, filter=get_variable('w2', [3, 3, net5_kernal_size_1, net5_kernal_size_2]),
                           strides=[1, 1, 1, 1],
                           padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b2', [net5_kernal_size_2]))
        net = tf.nn.relu(net)
    # maxpool
    with tf.variable_scope('net6'):
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # conv3-512 conv3-512
    with tf.variable_scope('net7'):
        net = tf.nn.conv2d(net, filter=get_variable('w1', [3, 3, net5_kernal_size_2, net7_kernal_size_1]),
                           strides=[1, 1, 1, 1],
                           padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b1', [net7_kernal_size_1]))
        net = tf.nn.relu(net)

        net = tf.nn.conv2d(net, filter=get_variable('w2', [3, 3, net7_kernal_size_1, net7_kernal_size_2]),
                           strides=[1, 1, 1, 1],
                           padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b2', [net7_kernal_size_2]))
        net = tf.nn.relu(net)
    # maxpool
    with tf.variable_scope('net8'):
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # conv3-512 conv3-512
    with tf.variable_scope('net9'):
        net = tf.nn.conv2d(net, filter=get_variable('w1', [3, 3, net7_kernal_size_2, net9_kernal_size_1]),
                           strides=[1, 1, 1, 1],
                           padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b1', [net9_kernal_size_1]))
        net = tf.nn.relu(net)

        net = tf.nn.conv2d(net, filter=get_variable('w2', [3, 3, net9_kernal_size_1, net9_kernal_size_2]),
                           strides=[1, 1, 1, 1],
                           padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b2', [net9_kernal_size_2]))
        net = tf.nn.relu(net)
    # maxpool
    with tf.variable_scope('net10'):
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # fc
    with tf.variable_scope('net11'):
        # 将四维的数据转换为两维的数据
        shape = net.get_shape()
        feature_number = shape[1] * shape[2] * shape[3]
        net = tf.reshape(net, shape=[-1, feature_number])
        # 全连接
        net = tf.add(tf.matmul(net, get_variable('w', [feature_number, net11_unit_size])),
                     get_variable('b', [net11_unit_size]))
    # fc
    with tf.variable_scope('net12'):
        # 全连接
        net = tf.add(tf.matmul(net, get_variable('w', [net11_unit_size, net12_unit_size])),
                     get_variable('b', [net12_unit_size]))
    # fc
    with tf.variable_scope('net13'):
        # 全连接
        net = tf.add(tf.matmul(net, get_variable('w', [net12_unit_size, net13_unit_size])),
                     get_variable('b', [net13_unit_size]))

    # softmax
    with tf.variable_scope('net14'):
        # softmax
        act = tf.nn.softmax(net)

    return act


# 获取网络
act = vgg_network(x, y)

# 构建损失函数，优化器，准确率评估
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=act))  # 损失函数

# 优化器
# AdamOptimizer通过使用动量（参数的移动平均数）来改善传统梯度下降，促进超参数动态调整。
# 我们可以通过创建标签错误率的摘要标量来跟踪丢失和错误率
# 一个寻找全局最优点的优化算法，引入了二次方梯度校正。
# 相比于基础SGD算法，1.不容易陷于局部优点。2.速度更快
train = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost)

# 正确率
# y [[1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],[0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]] ->arg_max(y,1)->[0,1]
# net  y [[1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0],[0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0] ]
correct_pred = tf.equal(tf.argmax(y, axis=1), tf.argmax(act, axis=1))
acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 训练
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    # 2.初始化所有的变量（一定在变量构建之后调用初始化操作）
    tf.global_variables_initializer().run()
    # 3.迭代训练
    for epoch in range(train_epoch):
        # 4.计算一次迭代batch执行的次数
        total_batch = int(total_sample_number / batch_size) - 5
        # 迭代更新
        for step in range(total_batch):
            # 获取当前批次的数据
            train_x = X[step * batch_size:step * batch_size + batch_size]
            train_y = Y[step * batch_size:step * batch_size + batch_size]
            # 模型训练
            sess.run(train, feed_dict={x: train_x, y: train_y})

            # 每更新10次，输出一下
            if step % 10 == 0:
                loss, accuracy = sess.run([cost, acc], feed_dict={x: train_x, y: train_y})
                print('迭代次数:{}, 步骤：{}, 训练集损失函数：{}, 训练集准确率：{}'.format(epoch, step, loss, accuracy))

        # 展示一下
        if epoch % 2 == 0:
            # 获取测试集数据
            test_x = X[step * batch_size:]
            test_y = Y[step * batch_size:]
            loss, accuracy = sess.run([cost, acc], feed_dict={x: test_x, y: test_y})
            print('*' * 100)
            print('步骤:', epoch)
            print('测试集损失函数值：{}，测试集准确率：{}'.format(loss, accuracy))
            loss, accuracy = sess.run([cost, acc], feed_dict={x: train_x, y: train_y})
            print('训练集损失函数：{}, 训练集准确率：{}'.format(loss, accuracy))

    # TODO: 自己在这个代码中加入，模型保存的内容，以及可视化相关的内容

print("End！！")
