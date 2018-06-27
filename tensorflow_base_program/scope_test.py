import tensorflow as tf

# 减少同类变量定义的案例

## 不用 variable_scope的情况
def my_func(x):
    weight1 = tf.Variable(tf.random_normal([1]))
    bias1 = tf.Variable(tf.random_normal([1]))
    result1 = weight1*x + bias1

    weight2 = tf.Variable(tf.random_normal([1]))
    bias2 = tf.Variable(tf.random_normal([1]))
    result2 = weight1*x + bias1

    return result1, weight1, bias1, result2, weight2, bias2
    # weight1 = tf.get_variable('weight', [1], initializer=tf.random_normal_initializer())
    # weight2 = tf.get_variable('weight', [1], initializer=tf.random_normal_initializer())

init_op = my_func(tf.constant([3], dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(init_op))

print('=================下面是简化方法，可以实现变老了共享===================')

def my_func2(x):
    weight = tf.get_variable('weight', [1], initializer=tf.random_normal_initializer())
    bias = tf.get_variable('bias', [1], initializer=tf.random_normal_initializer())
    result = weight * x + bias

    return result, weight, bias

def func(x):
    with tf.variable_scope('op1'): # 指定了变量不同的命令空间
        r1 = my_func2(x)
    with tf.variable_scope('op2'):
        r2 = my_func2(x)
    return r1, r2

new_op = func(tf.constant(3.0, dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(new_op))
    print(new_op)
    print(new_op[0][0].name) # op1/add:0   表示了不同的命名空间
    print(new_op[1][0].name) #  op2/add:0

print('==============tf.variable_scope====================')
# 这一段尽然没有报错 无法理解啊！！！
'''
a = tf.Variable([3], name='aa')
b = tf.Variable([3], name='aa')
c = tf.multiply(a, b)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(c))s
'''
###  reuse False:必须这里创建，  True: 这里一定不能创建 tf.AUTO_REUSE: 有就共享，没有就创建
with tf.variable_scope('foo', initializer=tf.constant_initializer(3.0), reuse=tf.AUTO_REUSE):
    vv = tf.get_variable('vv', [2])  # [2] 表示 shape

with tf.variable_scope('bar', initializer=tf.random_normal_initializer()):
    kk = tf.get_variable('kk', [1])

with tf.variable_scope('foo',  reuse=True): ## 此处实现了变量的共享
    ww = tf.get_variable('vv', [2])  # [2] 表示 shape

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(vv))
    print(vv.name)

    print(sess.run(ww))  ## ww 和 vv 的值是一样的
    print(ww.name)
    if ww == vv:
        print("ww==vv")


# 这里是老师写的代码
with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    with tf.variable_scope('foo', initializer=tf.constant_initializer(4.0)) as foo:
        v = tf.get_variable("v", [1])
        w = tf.get_variable("w", [1], initializer=tf.constant_initializer(3.0))
        with tf.variable_scope('bar'):
            l = tf.get_variable("l", [1]) # 使用foo的 initializer

            with tf.variable_scope(foo):  # 此处的命名空间与上面的foo的共享变量
                h = tf.get_variable('h', [1]) # 使用foo的 initializer
                g = v + w + l + h

    with tf.variable_scope('abc'):
        a = tf.get_variable('a', [1], initializer=tf.constant_initializer(5.0))
        b = a + g

    sess.run(tf.global_variables_initializer())
    # 深刻理解下面的内容
    print("{},{}".format(v.name, v.eval()))
    print("{},{}".format(w.name, w.eval()))
    print("{},{}".format(l.name, l.eval()))
    print("{},{}".format(h.name, h.eval()))
    print("{},{}".format(g.name, g.eval()))
    print("{},{}".format(a.name, a.eval()))
    print("{},{}".format(b.name, b.eval()))

print('========================the difference of name_scope and variable_scope=============================')
## name_scope 与 variable_scope 的区别
with tf.Session() as sess:
    with tf.name_scope('name1'):
        with tf.variable_scope('variable1'):
            v = tf.Variable(1.0, name='v')
            w = tf.get_variable(name='w', shape=[1], initializer=tf.constant_initializer(2.0))
            h = w + v

    with tf.variable_scope('variable2'):
        with tf.name_scope('name2'):
            v2 = tf.Variable(2.0, name='v2')
            w2 = tf.get_variable(name='w2', shape=[1], initializer=tf.constant_initializer(2.0))
            h2 = v2 + w2

    sess.run(tf.global_variables_initializer()) # 必须放在后面
    print("{},{}".format(v.name, v.eval()))
    print("{},{}".format(w.name, w.eval()))
    print("{},{}".format(h.name, h.eval()))
    print("{},{}".format(v2.name, v2.eval()))
    print("{},{}".format(w2.name, w2.eval()))
    print("{},{}".format(h2.name, h2.eval()))












