import  tensorflow as tf
# 可视化
with tf.variable_scope("foo"):
    with tf.device("/cpu:0"):
        x_init1 = tf.get_variable('init_x', [10], tf.float32, initializer=tf.random_normal_initializer())[0]
        x = tf.Variable(initial_value=x_init1, name='x')
        y = tf.placeholder(dtype=tf.float32, name='y')
        z = x + y

    # update x
    assign_op = tf.assign(x, x + 1)
    with tf.control_dependencies([assign_op]):
        with tf.device('/gpu:0'):
            out = x * y

with tf.device('/cpu:0'):
    with tf.variable_scope("bar"):
        a = tf.constant(3.0) + 4.0
    w = z * a

# 开始记录信息(需要展示的信息的输出)
tf.summary.scalar('scalar_init_x', x_init1)
tf.summary.scalar(name='scalar_x', tensor=x)
tf.summary.scalar('scalar_y', y)
tf.summary.scalar('scalar_z', z)
tf.summary.scalar('scala_w', w)
tf.summary.scalar('scala_out', out)

with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    # merge all summary
    merged_summary = tf.summary.merge_all()
    # 得到输出到文件的对象
    writer = tf.summary.FileWriter('./result', sess.graph) # 只需要图  就这一句就可以了

    # 初始化
    sess.run(tf.global_variables_initializer())
    # print
    for i in range(1, 5):
        summary, r_out, r_x, r_w = sess.run([merged_summary, out, x, w], feed_dict={y: i})
        writer.add_summary(summary, i)
        print("{},{},{}".format(r_out, r_x, r_w))

    # 关闭操作
    writer.close()
