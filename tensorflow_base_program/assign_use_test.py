import tensorflow as tf

# tf 实现一个累加器
# 1.创建变量
x = tf.Variable(0, dtype=tf.int32, name='x')
# x = x + 1

# 2. 变量的更新     tf.assign(A, new_number): 这个函数的功能主要是把A的值变为new_number
assign_op = tf.assign(ref= x,  value=x+1)

# 2. 全局变量的初始化
init_op = tf.global_variables_initializer()

# 3. 创建Session 执行graph
with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    sess.run(init_op)
    for i in range(5):
        print(sess.run(assign_op))

##  动态的更新变量的维度数目
x = tf.Variable(initial_value=[],
                dtype=tf.float32,
                trainable=False,
                validate_shape=False # 设置为True，表示在变量更新的时候，进行shape的检查，默认为True
                )
concat = tf.concat([x, [0.0, 0.0]], axis=0)    # 不能是  [0, 0]
assign_op2 =  tf.assign(x, concat, validate_shape=False)

## 变量初始化操作
x_init_op = tf.global_variables_initializer()

# 创建session，执行变量

with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    sess.run(x_init_op)
    for i in range(5):
        print(sess.run(assign_op2))
        sess.run(concat)

### 求阶乘
# 定义一个变量
sum = tf.Variable(1, dtype=tf.int32)

# 定义一个占位符
i= tf.placeholder(dtype=tf.int32)

# 3. 更新操作
tmp_sum = sum * i
# tmp_sum = tf.multiply(sum, i)
assign_op = tf.assign(sum, tmp_sum)
with tf.control_dependencies([assign_op]):
    # 如果需要执行这个代码块中的内容，必须先执行control_dependencies中给定的操作/tensor
    sum = tf.Print(sum, data=[sum, sum.read_value()], message='sum:')

# 4. 变量初始化操作
x_init_op = tf.global_variables_initializer()

# 5. 运行
with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    # 变量初始化
    sess.run(x_init_op)

    # 模拟迭代更新累加器
    for j in range(1, 6):
        # 执行更新操作
        # sess.run(assign_op, feed_dict={i: j})
        # 通过control_dependencies可以指定依赖关系，这样的话，就不用管内部的更新操作了
        r = sess.run(sum, feed_dict={i: j})

    print("5!={}".format(r))


