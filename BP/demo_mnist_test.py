import tensorflow as tf
import matplotlib as mpl
from  tensorflow.examples.tutorials.mnist import input_data

# 设置字符集
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 数据加载
mnist = input_data.read_data_sets('data/', one_hot=True)

# 构建神经网络（4层，1 put , 2 hidden, 1 output）
n_unit_hidden_1 = 256 # 第一次层hidden中的神经元数目
n_unit_hidden_2 = 128 # 第二层的hidden中的神经元数目
n_input = 784 # 输入一个样本（图像）是28*28 像素的
n_classes = 10 # 输出的类别数目

#　定义输入的占位符
x = tf.placeholder(tf.float32, shape=[None, n_input], name='x')
y = tf.placeholder(tf.float32, shape=[None, n_classes], name='y')

# 构建初始化的w 和 b
weights = {
    'w1': tf.Variable(tf.random_normal(shape=[n_input, n_unit_hidden_1], stddev=0.1)),
    'w2': tf.Variable(tf.random_normal(shape=[n_unit_hidden_1, n_unit_hidden_2], stddev=0.1)),
    'out': tf.Variable(tf.random_normal(shape=[n_unit_hidden_2, n_classes], stddev=0.1))
}

biases = {
    'b1': tf.Variable(tf.random_normal(shape=[n_unit_hidden_1], stddev=0.1)),
    'b2': tf.Variable(tf.random_normal(shape=[n_unit_hidden_2], stddev=0.1)),
    'out': tf.Variable(tf.random_normal(shape=[n_classes], stddev=0.1))
}

def multiplayer_perceotron(_X, _weights, _biases):
    # 第一层 -》第二层 input -> hidden1
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['w1']), _biases['b1']))
    # 第二层 -》 第三层 hidden1 -》 hidden2
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, _weights['w2']), _biases['b2']))
    # 第三层 -》 第四层  hidden2 -> output
    return tf.matmul(layer2, _weights['out']) + _biases['out']  # 返回值没有激活

# 获取预测值
act = multiplayer_perceotron(x, weights, biases)

# 构建模型的损失函数
# softmax_cross_entropy_with_logits: 计算softmax中的每个样本的交叉熵，logits指定预测值，labels指定实际值
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=act, labels=y))

# 使用梯度下降求解
# 使用梯度下降，最小化误差
# learning_rate: 要注意，不要过大，过大可能不收敛，也不要过小，过小收敛速度比较慢
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# 得到预测的类别是那一个
# tf.argmax:对矩阵按行或列计算最大值对应的下标，和numpy中的一样
# tf.equal:是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的
pred = tf.equal(tf.argmax(act, axis=1), tf.argmax(y, axis=1))
# 正确率（True转换为1，False转换为0）
acc = tf.reduce_mean(tf.cast(pred, tf.float32))

# 初始化
init = tf.global_variables_initializer()

# 执行模型的训练
batch_size = 100  # 每次处理的图片数
display_step = 4  # 每4次迭代打印一次
# LAUNCH THE GRAPH
with tf.Session() as sess:
    # 进行数据初始化
    sess.run(init)

    # 模型保存、持久化
    saver = tf.train.Saver()
    epoch = 0
    while True:
        avg_cost = 0
        # 计算出总的批次
        total_batch = int(mnist.train.num_examples / batch_size)
        # 迭代更新
        for i in range(total_batch):
            # 获取x和y
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feeds = {x: batch_xs, y: batch_ys}
            # 模型训练
            sess.run(train, feed_dict=feeds)
            # 获取损失函数值
            avg_cost += sess.run(cost, feed_dict=feeds)

        # 重新计算平均损失(相当于计算每个样本的损失值)
        avg_cost = avg_cost / total_batch

        # DISPLAY  显示误差率和训练集的正确率以此测试集的正确率
        if (epoch + 1) % display_step == 0:
            print("批次: %03d 损失函数值: %.9f" % (epoch, avg_cost))
            feeds = {x: mnist.train.images, y: mnist.train.labels}
            train_acc = sess.run(acc, feed_dict=feeds)
            print("训练集准确率: %.3f" % train_acc)
            feeds = {x: mnist.test.images, y: mnist.test.labels}
            test_acc = sess.run(acc, feed_dict=feeds)
            print("测试准确率: %.3f" % test_acc)

            if train_acc > 0.9 and test_acc > 0.9:
                saver.save(sess, './mn/model')
                break
        epoch += 1

    # 模型可视化输出
    writer = tf.summary.FileWriter('./mn/graph', tf.get_default_graph())
    writer.close()
