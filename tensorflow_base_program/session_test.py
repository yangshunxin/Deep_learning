import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 消除警告


a = tf.constant([1, 2], dtype=tf.int32, shape=[1, 2])
b = tf.constant([[1, 2],[3, 4]], dtype=tf.int32, shape=[2, 2])
c = tf.matmul(a, b)

## 启动默认图
# target='', graph=None, config=None   三个参数--有意思
sess = tf.Session(graph=tf.get_default_graph())

# # 调用sess的run方法来执行矩阵的乘法，得到c的结果值（所以将c作为参数传递进去）
# # 不需要考虑图中间的运算，在运行的时候只需要关注最终结果对应的对象以及所需要的输入数据值
# # 只需要传递进去所需要得到的结果对象，会自动的根据图中的依赖关系触发所有相关的OP操作的执行
# # 如果op之间没有依赖关系，tensorflow底层会并行的执行op(有资源) --> 自动进行
# # 如果传递的fetches是一个列表，那么返回值是一个list集合
# # fetches：表示获取那个op操作的结果值
result = sess.run(fetches=[a, c])  # result类型为 <class 'numpy.ndarray'>
print(result)
# 任务关闭
sess.close()
# 会报错 关闭后不能再用了
# sess.run(c)

# 用with模块来自动关闭
with tf.Session() as sese2:
    # result2 = sese2.run(c) # 可以传入fetches 参数
    result2 = c.eval()  # 这一句与上面一句有相同的作用，直接返回c的值
    print(result2)

# 对config中参数的设置
a  = tf.constant('10', tf.string, name='a_const')
b = tf.string_to_number(a, out_type=tf.float64, name='str_2_double')
c = tf.to_double(5.0, name='to_double')
d = tf.add(b, c, name='add')

# 构建Session并执行图
# 1. 构建GPU 相关参数
gpu_options = tf.GPUOptions()
# per_process_gpu_memory_fraction：给定对于每一个进程，分配多少的GPU内存，默认是1
# 设置为0.5 表示分配50%的GPU内存
gpu_options.per_process_gpu_memory_fraction = 0.5

# allow_growth: 设置为True表示在进行GPU内存分配的时候，采用动态分配方式，默认为False
# 动态分配的意思是指，在启动之前，不分配全部内存，根据需要，后面动态的进行内存分配，
# 在启动动态分配后，GPU内存不会自动释放（故：复杂、长时间运行的任务不建议设置为True）， --False时会自动释放
gpu_options.allow_growth = True

#2. 构建Graph优化的相关参数
optimizer = tf.OptimizerOptions(
    do_common_subexpression_elimination = True, # 设置为True表示开启公共执行子语句
    do_constant_folding=True, # 设置为True表示开始常数折叠优化
    opt_level = 0 # 设置为0 表示开始上述两个优化，默认就是0
)

graph_options = tf.GraphOptions(optimizer_options=optimizer)

# 3. 构建Session的Config
# allow_soft_placement：是否允许动态使用CPU和GPU，默认为False；
    # 当我们的安装方式为GPU时，建议设置该参数为True，因为TensorFlow中的部分op只能在CPU上运算
# log_device_placement：是否打印日志，默认为False，不打印日志
config_proto = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True,
                              graph_options=graph_options, gpu_options=gpu_options)

# 4. 构建Session并运行
with tf.Session(config=config_proto) as sess:
    print(sess.run(d))
    print(d.eval()) # 用Tensro.eval() 来执行session

# 进入交互式会话
sess = tf.InteractiveSession()

# 定义变量和常量
x = tf.constant([1.0, 2.0])
a = tf.constant([2.0, 4.0])
# 进行减操作
sub = tf.subtract(x, a)

# 输出结果
print(sub.eval())
print(sess.run(sub))