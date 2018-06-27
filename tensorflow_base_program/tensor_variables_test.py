import tensorflow as tf

a = tf.constant([3, 9, 0]) # <class 'tensorflow.python.framework.ops.Tensor'>
print(type(a))
print(a)

# 创建一个变量，初始化值为标量 3.0
a = tf.Variable(3.0) # 必须初始化

# 创建常量
b = tf.constant(2.0)
c = tf.add(a, b)

# 启动图后，变量必须先经过初始化操作
# 增加一个初始化变量的op到图中
# tf.initialize_all_variables(): 初始化全局所有变量
init_op = tf.global_variables_initializer()
# 相当于在图中加入一个初始化全局变量的操作

# 启动图
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
    # 运行 init op
    #  添加一个操作
    sess.run(init_op)
    # 这行代码也是初始化运行操作，但是要求明确给定当前代码块对应的默认session(tf.get_default_session())是哪个，底层使用默认session来运行
    # init_op.run()
    # 获取值
    print("a={}".format(sess.run(a)))
    print("c={}".format(c.eval()))

'''
# 初始化时变量依赖的案例----在1.4.0版本后就没有这个问题
# 创建一个变量
# random_normal(shape,mean=0.0, stddev=1.0, dtype=dtypes.float32,seed=None,name=None)
# 第一个参数时shape， [10]表示 10行一列的数组 mean：表示均值， stddev:表示方差，
w1 = tf.Variable(tf.random_normal([10], stddev=0.5, dtype=tf.float32), name='w1')

# 基于第一个变量创建第二个变量 ？？？？？？？？？？？？？？？？
a = tf.constant(2, dtype=tf.float32)
# 这里要注意了，要用initial_value
# w2 = tf.Variable(w1.initial_value()*a, name='w2')  # 报错： 'Tensor' object is not callable
w2 = tf.Variable(w1*a, name='w2') # 这里要注意了， #  报错：Attempting to use uninitialized value w2

# 变量全局初始化 （上面有了）


#启动图
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    # 运行init_op
    sess.run(init_op)
    # 获取值
    result = sess.run([w1, w2])
    print(result)
'''

### feed 测试
# 创建占位符，创建图
m1 = tf.placeholder(tf.float32)
m2 = tf.placeholder(tf.float32)
m3 = tf.placeholder_with_default(4.0, shape=None)
output = tf.multiply(m1, m2) # 乘法
ot1 = tf.add(m1, m3)

# 运行图
with tf.Session() as sess:
    print(sess.run(output, feed_dict={m1:3, m2: 4}))
    print(output.eval(feed_dict={m1:8, m2:5}))

