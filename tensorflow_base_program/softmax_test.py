'''
softmax 用于二分类
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import Binarizer, OneHotEncoder
# 下面时总纲领：
# softmax做多分类原理就是 构建与样本类别相同个数的回归器，输出值按照softmax函数来处理，得到样本在每个分类器上的概率值 ，
# 预测时 测试样本通过所有的分类器，计算出在每个分类器上的值，并做概率计算

# 1. 模拟数据产生
np.random.seed(28)
n = 100
x_data = np.random.normal(loc=0, scale=2, size=(n,2)) # (100,2)
y_data = np.dot(x_data,np.array([[5], [3]])) # (2, 1)
# 成为了 (100, 2) 每行只有一个1
y_data = OneHotEncoder().fit_transform(Binarizer(threshold=0).fit_transform(y_data)).toarray()


# 构建最终画图的数据
t1 = np.linspace(-8, 10, 100)
t2 = np.linspace(-8, 10, 100)
xv, yv = np.meshgrid(t1, t2)
x_test = np.dstack((xv.flat, yv.flat))[0]


# 2. 模型构建
# 构建数据输入占位符x和y
# x/y: 行表示样本个数，用None表示个数不定，
# x: 2表示输入样本的特征个数，用列表示
# y: 2表示哑编码后的类型个数
x = tf.placeholder(tf.float32, [None, 2], name='x')
y = tf.placeholder(tf.float32, [None, 2], name='y')

# 构建权重w和偏执值b
# w中第一个2表示样本的分类个数，即回归模型的个数， 第二个2表示样本的特征个数
# b中2是样本的分类个数
w = tf.Variable(tf.zeros([2,2]), name='w')
b = tf.Variable(tf.zeros([2]), name='b') # （1， 2）
# tf.matmul(x, w) + b 的行数不定，列数为2， 两列的值分别为两个模型的预测值，行表示样本
# softmax 其实就是把每一行的两个数据，做了一个转换，转换成了概率值
#act(Tensor)是 行数由样本个数决定，有2列，一行中的每列的两个值是概率值，和为1
act = tf.nn.softmax(tf.matmul(x, w) + b)

# 构建模型的损失函数
# tf.reduce_mean(x， axis=0) 求x中所有值的均值, 当axis= 0 表示求列的均值，，axis=1表示求行的均值
cost = -tf.reduce_mean(tf.reduce_mean(y * tf.log(act), axis=1)) # 其实多除了2

# 使用梯度下降法，求解最小误差
# learning_rate 过大会不收敛，，过小收敛速度慢
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# 得到预测的类别是那一个
# tf.argmax:对矩阵按行或列计算最大值对应的下标，和numpy中的一样
# tf.equal:是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的
pred = tf.equal(tf.argmax(act, axis=1), tf.argmax(y, axis=1))
# 正确率（True转换为1，False转换为0）  均值就是正确率
acc = tf.reduce_mean(tf.cast(pred, tf.float32))

# 初始化
init = tf.global_variables_initializer()

# 总共训练迭代次数
training_epochs = 50
# 批次数量  一次10个样本， 总共n个样本
num_batch = int(n/10)
# 训练迭代次数（打印信息）
display_step = 5

with tf.Session() as sess:
    # 变量初始化
    sess.run(init)

    for epoch in range(training_epochs): # 迭代次数
        # 做一次迭代操作
        avg_cost = 0
        # 打乱数据次序
        index = np.random.permutation(n) # n是总样本数
        for i in range(num_batch): # 批量梯度时，每一个批次
            # 获取传入进行模型训练的10个数据对应的索引
            xy_index = index[i * 10:(i + 1)* 10]
            # 构建按传入的feed 参数
            feeds = {x:x_data[xy_index], y:y_data[xy_index]}
            # 进行模型训练
            sess.run(train, feed_dict=feeds)
            # 可选：获取损失函数值---木有用
            avg_cost += sess.run(cost, feed_dict=feeds)/num_batch

        if epoch %  display_step == 0:
            feeds_train = {x: x_data, y: y_data}
            train_acc = sess.run(acc, feed_dict=feeds_train)
            print("迭代次数: %03d/%03d 损失值: %.9f 训练集上准确率: %.3f" % (epoch, training_epochs, avg_cost, train_acc))

    # 对用于画图的数据进行预测
    # y_hat: 是一个None*2的矩阵
    y_hat = sess.run(act, feed_dict={x: x_test})
    # 根据softmax分类的模型理论，获取每个样本对应出现概率最大的(值最大的)
    # y_hat：是一个None*1的矩阵
    y_hat = np.argmax(y_hat, axis=1)

print("模型训练完成")
# 画图展示一下
cm_light = mpl.colors.ListedColormap(['#bde1f5', '#f7cfc6'])
y_hat = y_hat.reshape(xv.shape)
plt.pcolormesh(xv, yv, y_hat, cmap=cm_light)  # 预测值
plt.scatter(x_data[y_data[:, 0] == 0][:, 0], x_data[y_data[:, 0] == 0][:, 1], s=50, marker='+', c='red')
plt.scatter(x_data[y_data[:, 0] == 1][:, 0], x_data[y_data[:, 1] == 0][:, 1], s=50, marker='o', c='blue')
plt.show()





