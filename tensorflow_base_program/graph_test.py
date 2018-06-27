import tensorflow as tf

## 对图的测试
# 创建一个常量op，产生一个1x2的矩阵，这个op被作为一个节点----源节点
# 默认是加入到默认图中
# 构造器的返回值代表该常量op的返回值
mat1 = tf.constant([3, 3], dtype=tf.float32, shape=[1, 2])
print(type(mat1)) # 是一个Tensor对象  <class 'tensorflow.python.framework.ops.Tensor'>

# 创建另一个常量op，产生要给2x1的矩阵
mat2 = tf.constant([[2.],[2.]], dtype=tf.float32, shape=[2, 1])

# 创建一个矩阵乘法op， 将mat1和mat2作为输入
# 返回值代表乘法op的结果
product = tf.matmul(mat1, mat2)

a = tf.constant(7.0)

print("变量mat1定义在默认图上：{}".format(mat1.graph is tf.get_default_graph()))

# 可以构建多个图
# 明确指定一个新图
g = tf.Graph()
with g.as_default(): # 不同的图
    #定义一个新的操作在图g上
    b = tf.constant(3.0)
    c = tf.constant(5.0)
    print("变量b定义在图g上：{}".format(b.graph is g))
    print("变量c定义在图g上：{}".format(c.graph is g))
    print("当前的默认graph是b的图吗：{}".format(b.graph is tf.get_default_graph()))

# 这一句会报错，不同图中的节点不能相连
# d = tf.matmul(a, b)