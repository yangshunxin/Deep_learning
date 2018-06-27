import tensorflow as tf

# # 模型保存
# v1 = tf.Variable(tf.constant(3.0), name='v1')
# v2 = tf.Variable(tf.constant(4.0), name='v2')
# result = tf.add(v1, v2, name='v1_v2_add')
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     # 保存模型到model文件夹下，文件前缀为：model.ckpt
#     saver.save(sess, './model/model.ckpt')

# # 模型完整提取（完整提取：需要完整恢复保存之前的数据格式, 模型定义必须一模一样，包括变量名称）
# v1 = tf.Variable(tf.constant(4.0), name='v1')
# v2 = tf.Variable(tf.constant(4.0), name='v2')
# result = tf.add(v1, v2, name='v1_v2_add')
#
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     # 会从对应的文件夹中加载 变量、图等相关信息
#     saver.restore(sess, './model/model.ckpt')
#     print(sess.run([result]))

# # 直接加载图，不需要定义变量了
# saver = tf.train.import_meta_graph('./model/model.ckpt.meta')
#
# with tf.Session() as sess:
#     saver.restore(sess, './model/model.ckpt')
#     print(sess.run(tf.get_default_graph().get_tensor_by_name('v1_v2_add:0'))) # 也是可以调用的

# 当变量名修改后，加载适应
a = tf.Variable(tf.constant(4.0), name='a')
b = tf.Variable(tf.constant(4.0), name='b')
result = tf.add(a, b, name='v1_v2_add')

saver = tf.train.Saver({'v1':a, 'v2':b})
with tf.Session() as sess:
    saver.restore(sess, './model/model.ckpt')
    print(sess.run(result)) # 7.0
