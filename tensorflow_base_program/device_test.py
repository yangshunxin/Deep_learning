import  tensorflow as tf

with tf.device('/cpu:0'):
    # 这个代码块中定义的操作，会在tf.device给定的设备上运行
    # 有一些操作，是不会再GPU上运行的（一定要注意）
    # 如果按照的tensorflow cpu版本，没法指定运行环境的
    a = tf.Variable([1, 2, 3], dtype=tf.int32, name='a')
    b = tf.constant(2, dtype=tf.int32, name='b')
    c = tf.add(a, b, name='ab')

with tf.device('/gpu:0'):
    # 这个代码块中定义的操作，会在tf.device给定的设备上运行
    # 有一些操作，是不会再GPU上运行的（一定要注意）
    # 如果按照的tensorflow cpu版本，没法指定运行环境的
    d = tf.Variable([2, 8, 13], dtype=tf.int32, name='d')
    e = tf.constant(2, dtype=tf.int32, name='e')
    f = d + e

g = c + f
# allow_soft_placement  含义非常重要
with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    # 初始化
    tf.global_variables_initializer().run()
    # 执行结果
    print(g.eval())


