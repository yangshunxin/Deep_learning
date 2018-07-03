import tensorflow as tf
from model import generator, discriminator
import numpy as np
import math
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

mnist = read_data_sets('data/', one_hot=True)
train_mnist = mnist.train.images
test_mnist = mnist.test.images
train_x = np.concatenate([train_mnist, test_mnist], 0)

train_label = mnist.train.labels
test_label = mnist.test.labels

train_y = np.concatenate([train_label, test_label], 0)
BATCH_SIZE = 64
images = tf.placeholder(tf.float32, [BATCH_SIZE, 28, 28, 1])
z = tf.placeholder(tf.float32, [BATCH_SIZE, 100])
y = tf.placeholder(tf.float32, [BATCH_SIZE, 10], name='y')

# 分类因素
y_dim = 10  # 分类的输出
c_dim = 1  # d的softmax输出
img_dim = 1  #

# 生成模型
G = generator(z, y)
# 判别模型
# 真实图传入判别模型进行训练
D, D_logits = discriminator(images, y)
# 传入生成的图进入判别模型
D_, D_logits_ = discriminator(G, y, reuse=True)

# 损失函数
# 生成模型的损失
# ones_like 根据传入的数据的形状，建立一个1的矩阵
# 希望生成模型能生成出 判别模型判别数据 真实概率为1的数据
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.ones_like(D_)))

# 判别模型的损失
# 希望判别模型能把真实数据判别为1
d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits, labels=tf.ones_like(D)))
# 希望判别模型能把生成模型生成的数据判别为0
d_fack_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.zeros_like(D_)))
d_loss = d_fack_loss + d_real_loss

# summary
# .......
# 优化
d_opt = tf.train.GradientDescentOptimizer(0.1).minimize(d_loss)
g_opt = tf.train.GradientDescentOptimizer(0.1).minimize(g_loss)

# 训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(25):
        # total_batch
        total_batch_size = int(math.ceil(len(train_x) * 1.0 / BATCH_SIZE))
        for idx in range(total_batch_size):
            start_index = idx * BATCH_SIZE
            end_index = start_index + BATCH_SIZE
            X = train_x[start_index:end_index].reshape([-1, 28, 28, 1])
            Y = train_y[start_index:end_index]
            Z = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))

            # 更新d的参数
            feed_dict_d = {images: X, z: Z, y: Y}
            sess.run(d_opt, feed_dict=feed_dict_d)
            # 更新g的参数
            feed_dict_g = {z: Z, y: Y}
            sess.run(g_opt, feed_dict=feed_dict_g)

            # 计算损失
            errD_fake = d_fack_loss.eval(feed_dict_g)
            feed_dict_d_1 = {images: X, y: Y}
            errD_real = d_real_loss.eval(feed_dict_d_1)
            errG = g_loss.eval(feed_dict_g)

            if idx % 50 == 1:
                print('D loss', errD_fake, ',', errD_real)
                print('G loss', errG)
