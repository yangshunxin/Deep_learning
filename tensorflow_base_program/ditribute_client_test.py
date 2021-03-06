# -- encoding:utf-8 --
"""
分布式集群使用(最简单的)
Create by ibf on 2018/5/6
"""

import tensorflow as tf
import numpy as np

# 1. 构建图
with tf.device('/job:ps/task:0'): # 这里表示主机
    # 2. 构造数据
    x = tf.constant(np.random.rand(100).astype(np.float32))

# 3. 使用另外一个机器
with tf.device('/job:work/task:1'):
    y = x * 0.1 + 0.3

# 4. 运行  这里只用了一台机器？？？ target
with tf.Session(target='grpc://127.0.0.1:33335',
                config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    print(sess.run(y))
