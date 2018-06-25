import numpy as np
import  tensorflow as tf
import matplotlib.pyplot as plt

# tenforflow 实现线性回归

# 随机生成1000个点，围绕在y=0.1x+0.3的直线范围
num_points = 1000
vectors_set = []
for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55) # 生成一个均值为0，方差为0.55的高斯分布
    # x1 = i/100  # 无法收敛
    y1 = x1*0.1 + 0.3 + np.random.normal(0, 0.03) #wx+b 后面添加一些抖动
    vectors_set.append([x1, y1])

# 生成一些样本呢
x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

print(x_data[0:5])
print(y_data[0:5])

# 显示源数据
plt.scatter(x_data, y_data, c='r')
plt.show()

# 生成1维的w矩阵，取值是[-1, 1]之间的随机数
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='W')
# 生成1维的b矩阵，初始值是0
b = tf.Variable(tf.zeros([1]), name='b')
# 经过计算得出预估值y
y = W * x_data + b

# 以预估值y和实际值y_data之间的均方差作为损失
loss = tf.reduce_mean(tf.square(y-y_data), name='loss')
# 采用梯度下降法来优化参数
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 训练的过程就是最小化这个误差值
train = optimizer.minimize(loss, name='train')

sess = tf.Session()
# 全局变量 初始化
init = tf.global_variables_initializer()
sess.run(init)

# 初始化的w和b是多少
print('W==', sess.run(W), 'b=', sess.run(b), 'loss=', sess.run(loss))

# 执行20次训练
for step in range(20):
    sess.run(train)
    # 输出训练后的w和b
    print('W==', sess.run(W), 'b=', sess.run(b), 'loss=', sess.run(loss))

