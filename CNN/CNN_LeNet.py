import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/', one_hot=True)
trainimg   = mnist.train.images
trainlabel = mnist.train.labels
testimg    = mnist.test.images
testlabel  = mnist.test.labels
print ("MNIST ready")

# 28*28图片
n_input = 784
# 输出的大小
n_output = 10
# 权重  -----把计算过程搞清楚
weight = {
    'wc1':tf.Variable(tf.random_normal([5, 5, 1, 20], stddev=0.1)),
    'wc2':tf.Variable(tf.random_normal([5, 5, 20, 50], stddev=0.1)),
    'wd1':tf.Variable(tf.random_normal([7*7*50, 500], stddev=0.1)),
    'wd2':tf.Variable(tf.random_normal([500, n_output], stddev=0.1))
}

# 表数字  i 类的偏置量
biases = {
    'bc1':tf.Variable(tf.random_normal([20], stddev=0.1)),
    'bc2':tf.Variable(tf.random_normal([50], stddev=0.1)),
    'bd1':tf.Variable(tf.random_normal([500], stddev=0.1)),
    'bd2':tf.Variable(tf.random_normal([n_output], stddev=0.1)),
}

def conv_basic(_input, _w, _b, _keepratio):
    # [55000, 784]
    # INPUT, 转换矩阵形状，改成一个28*28*1的，厚度自动
    _input_r = tf.reshape(_input, shape=[-1, 28, 28, 1])  # [batch, in_height, in_width, in_channels]
    print('_input_r.shape=', _input_r.shape)
    # CONV LAYER 1
    # tf.nn.conv2d是TensorFlow里面实现卷积的函数
    # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
    # 除去name参数用以指定该操作的name，与方法有关的一共五个参数：
    # 第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，
        # 具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个4维的Tensor，要求类型为float32和float64其中之一
    # 第二个参数filter：相当于CNN中的卷积核，
        # 它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，
        # 具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，
        # 有一个地方需要注意，第三维in_channels，就是参数input的第四维
    # 第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4----!!!!
    # 第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，当其为‘SAME’时，表示卷积核可以停留在图像边缘
    # 第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true
    # 结果返回一个Tensor，这个输出，就是我们常说的feature map
    _conv1 = tf.nn.conv2d(_input_r, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')

    # tf.nn.relu：修正线性，max(feature, 0)
    # tf.nn.bias_add：这个函数的作用是将偏差项bias加到value上面
    # 这个操作你可以看做是 tf.add的一个特例，其中bias必须是一维的
    # 该API支持广播形式，因此value可以有任何维度
    # 但是，该API又不像tf.add，可以让bias的维度和value的最后一维不同
    _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1, _b['bc1']))

    # 最大池化
    # value：一个四维的Tensor。数据维度是[batch， height， width，channels]。数据类型是float32,float64,qint8,quint8,qint32。
    # ksize: 一个长度不小于4的整形数组。该参数指定滑动窗口再输入数据张量每一维上面的步长。
    # strides：一个长度不小于4的整形数组。该参数指定滑动窗口再输入数据张量每一维上面的步长。
    # padding：一个字符串，取值为 SAME 或者 VALID
    # name：（可选）为这个操作取要给名字
    _pool1 = tf.nn.max_pool(_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 处理过拟合操作  _keepratio 是dropout的比例
    _pool_dr1 = tf.nn.dropout(_pool1, _keepratio)

    # print (help(tf.nn.conv2d))
    # print (help(tf.nn.max_pool))

    # CONV LAYER2
    _conv2 = tf.nn.conv2d(_pool_dr1, _w['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2, _b['bc2']))
    _pool2 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    _pool_dr2 = tf.nn.dropout(_pool2, _keepratio)

    # VECTORIZE 向量化
    _dense1 = tf.reshape(_pool_dr2, [-1, _w['wd1'].get_shape().as_list()[0]])

    # FULLY CONNECTED LAYER 1
    _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
    _fc1_dr1 = tf.nn.dropout(_fc1, _keepratio)
    # FULLY CONNECTED LAYER 2
    _out = tf.add(tf.matmul(_fc1_dr1, _w['wd2']), _b['bd2'])
    # RETURN
    out = { 'input_r':_input_r,  'conv1': _conv1, 'pool1': _pool1, 'pool1_dr1': _pool_dr1,
            'conv2': _conv2, 'pool2': _pool2, 'pool_dr2': _pool_dr2, 'dense1': _dense1,
            'fc1': _fc1, 'fc_dr1': _fc1_dr1, 'out': _out
    }
    return out
print("CNN READY")

# tf.random_normal, 给出均值为mean，标准差为stdev的高斯随机数（场）
# 从服从指定正太分布的数值中取出指定个数的值   shape 为输出张量的形状
a = tf.Variable(tf.random_normal(shape=[3, 3, 1, 64], stddev=0.1)) #  shape
print(a)
a = tf.Print(a, [a], "a: ")
# # Vari的初始化  ---后面统一初始化
# init = tf.global_variables_initializer()
# # 建立会话
# sess = tf.Session()
# # 执行 初始化
# sess.run(init)
# sess.run(a)


# 通过操作符号变量来描述这些可交互的操作单元
# x不是一个特定的值， 而是一个占位符placeholder， 我们在TensorFlow运行计算时输入这个值。
# 我们希望能够输入任意数量的MNIST图像，每一张图展平成784维的向量。
# 我们用 2维的浮点数张量来表示这些图，这个张量的形状是[None, 784]。（这里的None表示此张量的第一个维度可以是任何长度的）
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
keepratio = tf.placeholder(tf.float32)

# FUNCTIONS

# 调用CNN函数，返回运算完的结果
_pred = conv_basic(x, weight, biases,keepratio)['out']

# 交叉熵
# 首先看输入logits，它的shape是[batch_size, num_classes]---batch：样本个数， num_classes：分类个数（one-hot编码）
# 一般来讲， 就是神经网络最后一层的输出z
# 另外一个输入是 labels，它的shape也是[batch_size, num_classes]， 就是我峨嵋你神经网络期望的输出。
# 这个函数的作用就是计算最后一层是softmax层的cross_entropy，只不过tensorflow把softmax计算与 cross entropy 计算放到一起了
# 用一个函数来实现，用来提高程序的运行速度
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=_pred,labels=y))

# Adam算法
# AdamOptimizer通过使用动量（参数的移动平均数）来改善传统梯度下降，促进超参数动态调整。
# 我们可以通过创建标签错误率的摘要标量来跟踪丢失和错误率
# 一个寻找全局最优点的优化算法，引入了二次方梯度校正
# 相比于基础 SGD算法，1.不容易陷入局部优点  2.速度更快
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# 比较
_corr = tf.equal(tf.arg_max(_pred, 1), tf.arg_max(y, 1))
#cast:将x或者x.values转换为dtype
#tf.reduce_mean  求tensor中平均值
#http://blog.csdn.net/lenbow/article/details/52152766
accr = tf.reduce_mean(tf.cast(_corr, tf.float32))

# 初始化
init = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()
# SAVER
print ("GRAPH READY")

sess = tf.Session()
sess.run(init)
sess.run(init_local)

# 训练次数
training_epochs = 15
# batch --一次训练的样本数
batch_size = 5
# 执行到第几次显示运行结果
display_step = 10
for epoch in range(training_epochs):  #迭代次数
    # 平均误差
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples/batch_size)
    # total_batch = 10
    # Loop over all batches 循环所有批次
    list1 = []
    ###   每次迭代都将所有的数据放到模型中训练，，，，
    ###  小批量迭代时将所有数据都放到模型中，不过每次只放预先定义好的一部分，分多次放入。
    for i in range(total_batch):  # 小批量梯度的次数
        #去除训练集合的下10条
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Fit training using batch data 使用批处理数据进行培训
        sess.run(optm, feed_dict={x: batch_xs, y: batch_ys, keepratio: 0.7})
        # Compute average loss 计算平均损失
        avg_cost += sess.run(cost, feed_dict={x:batch_xs, y:batch_ys, keepratio:1.0})/total_batch
        train_acc = sess.run(accr, feed_dict={x:batch_xs, y: batch_ys, keepratio:1.})

        # Display logs per epoch step 显示现在的状态
        if epoch % display_step == 0:
            print("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
            train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys, keepratio: 1.})

            print(" Training accuracy: %.3f" % (train_acc))
            test_acc = sess.run(accr, feed_dict={x: testimg, y: testlabel, keepratio: 1.})
            print(" Test accuracy: %.3f" % (test_acc))

print("OPTIMIZATION FINISHED")





