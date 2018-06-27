import tensorflow as tf
import numpy as np

np.random.seed(28)

# 分布式执行 回顾线性问题
# 1. 配置服务器信息---这些都是task
ps_hosts = ['127.0.0.1:33331', '127.0.0.1:33332']
work_hosts = ['127.0.0.1:33333', '127.0.0.1:33334', '127.0.0.1:33335']
# 创建集群
cluster = tf.train.ClusterSpec({'ps':ps_hosts, 'work':work_hosts})

# 2. 定义一些运行参数（运行该python文件时可以指定这些参数），
tf.app.flags.DEFINE_integer('task_index', default_value=0, docstring="Index of task with job")
FLAGS = tf.app.flags.FLAGS


# 3. 构建运行方法---这个是 Between-graph 异步，，每次计算都更新参数
def main(_):
    # 图的构建
    with tf.device(
            tf.train.replica_device_setter(worker_device='/job:work/task:%d' % FLAGS.task_index, cluster=cluster)
    ):     # worker_device的意思是指定到 work_device的task 0 上执行
        # 构建一个样本的占位符信息
        x_data = tf.placeholder(tf.float32, [10]) # shape为[10]
        y_data = tf.placeholder(tf.float32, [10])

        # 定义一个变量 w和b
        # 产生一个[-1.0, 1.0]之间的均匀分布的随机数
        w = tf.Variable(initial_value=tf.random_uniform(shape=[1], minval=-1.0, maxval=1.0), name='w')
        b = tf.Variable(initial_value=tf.zeros([1]), name='b') # b 为 0

        # 构建一个预测值
        y_hat = w*x_data + b

        # 构建一个损失函数 MSE   (预测值与实际值 之差的平方和 均值)
        loss = tf.reduce_mean(tf.square(y_hat-y_data), name='loss')

        global_step = tf.Variable(0, name='global_step', trainable=False)
        # 以随机梯度下降的方式优化损失函数
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
        # 在优化的过程中， 是让哪个函数最小化
        train = optimizer.minimize(loss, name='train', global_step=global_step)

    # 图的运行
        hooks = [tf.train.StopAtStepHook(10000000)] # 结束条件
        with tf.train.MonitoredTrainingSession( # 此处跟 tf.Session 不同
            master='grpc://' + work_hosts[FLAGS.task_index],
            is_chief=(FLAGS.task_index == 0), # 是否进行变量的初始化，设置为true，表示进行初始化
            checkpoint_dir='./tmp', # 保存临时信息
            save_checkpoint_secs=None,
            hooks=hooks  # 停止条件
        ) as mon_sess:      # 构建的是 MonitoredTrainingSession 而不是 tf.Session()
            while not mon_sess.should_stop():  # 直到停止
                # 每次都产生10个样本，  feed_dict进去，故无论是随机梯度、批量梯度、小批量梯度都是在这里修改
                N = 10
                train_x = np.linspace(0, 6, N) + np.random.normal(loc=0.0, scale=2, size=N)
                train_y = 14*train_x - 7 + np.random.normal(loc=0.0, scale=5.0, size=N)
                ## 将每次生成的样本给到图中执行，
                _, step, loss_v, w_v, b_v = mon_sess.run([train, global_step, loss, w, b],
                                                         feed_dict={x_data: train_x, y_data: train_y})
                if step % 100 == 0:
                    print('Step:{}, loss:{}, w:{}, b:{}'.format(step, loss_v, w_v, b_v))

if __name__== '__main__':
    tf.app.run()
