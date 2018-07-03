import tensorflow as tf
import math
import numpy as np
from tensorflow.contrib.layers.python.layers import batch_norm
batch_size=64
df_dim=64      # 判别的z
dfc_dim=1024   # 判别的fc
c_dim=1       # 判别模型输出大小
y_dim=10     # 分类
output_height=28 # 输出的高度
output_width=28 #  输出的宽度
gf_dim=64    # 生成的conv

gfc_dim=1024
# 随机权重
def weight(name,shape,stdev=0.02,trainable=True):
    dtype=tf.float32
    weights = tf.get_variable(name,shape,dtype=dtype,trainable=trainable,initializer=tf.random_normal_initializer(stddev=stdev,dtype=dtype))
    return weights
# 常数偏置量
def bias(name,shape,bias_start=0.0,trainable=True):
    dtype=tf.float32
    var=tf.get_variable(name,shape,dtype=dtype,trainable=trainable,initializer=tf.constant_initializer(bias_start,dtype=dtype))
    return var

# 全连接
def fully_connected(name,value,out_shape,with_w=False):
    with tf.variable_scope(name):
        shape=value.get_shape().as_list()
        weights=weight('weights_fc',[shape[1],out_shape],0.02)
        biases=bias('biases_fc',[out_shape],0.0)
        if with_w:
            return weights,biases,tf.matmul(value,weights)+biases
        else:
            return tf.matmul(value, weights) + biases

# 约束条件串联到特征图中
def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([
        x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w',[k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv
# 标准化
#decay 滑动平均的decay
#scale 如果为True则结果乘以gamma，否则不乘
#is_train 如果为真值，则表示训练过程，权重会更新，
# 否则表示测试过程，权重不动
def norm(name,value,is_train=True):
    with tf.variable_scope(name) as scope:
        if is_train:
            return batch_norm(value,decay=0.9,scale=True,epsilon=1e-5,is_training=is_train,scope=scope)
        else:
            return batch_norm(value, decay=0.9, scale=True, ellipsis=1e-5, is_training=is_train,reuse=True,scope=scope)

def lrelu(value):
    return tf.nn.leaky_relu(value)

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()
  with tf.variable_scope(scope or "Linear"):
      matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,tf.random_normal_initializer(stddev=stddev))
      bias = tf.get_variable("bias", [output_size],initializer=tf.constant_initializer(bias_start))
      if with_w:
          return tf.matmul(input_, matrix) + bias, matrix, bias
      else:
          return tf.matmul(input_, matrix) + bias
# 判别模型
# [///] is exist,..... resue =True
# 参数已经被设定过，是否重新载入相同的参数
def discriminator(image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()

        if not y_dim:
            h0 = lrelu(conv2d(image, df_dim, name='d_h0_conv'))
            h1 = lrelu(norm('d_bn1',conv2d(h0, df_dim * 2, name='d_h1_conv')))
            h2 = lrelu(norm('d_bn2',conv2d(h1, df_dim * 4, name='d_h2_conv')))
            h3 = lrelu(norm('d_bn3',conv2d(h2, df_dim * 8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [batch_size, -1]), 1, 'd_h4_lin')

            return tf.nn.sigmoid(h4), h4
        else:
            yb = tf.reshape(y, [batch_size, 1, 1, y_dim])
            x = conv_cond_concat(image, yb)
            h0 = lrelu(conv2d(x, c_dim + y_dim, name='d_h0_conv'))
            h0 = conv_cond_concat(h0, yb)

            h1 = lrelu(norm('d_bn1',conv2d(h0, df_dim + y_dim, name='d_h1_conv')))
            h1 = tf.reshape(h1, [batch_size, -1])
            h1 = tf.concat([h1, y], 1)
            h2 = lrelu(norm('d_bn2',linear(h1, dfc_dim, 'd_h2_lin')))
            h2 = tf.concat([h2, y], 1)

            h3 = linear(h2, 1, 'd_h3_lin')

            return tf.nn.sigmoid(h3), h3

    # 判别模型
    # y(2d)->reshape->yb(4d)
    # conv_cond_concat(x,yb)
    # (1)conv2d+lrelu->c1
    # conv_cond_concat(c1,yb)->h2
    # (2)lrelu(batch_norm(h2))->c2
    # conv_cond_concat(c2,yb)->h3
    # =>[batch,height,width,out_channel+10]
    # lrelu(norm(fc))->h4
    # fc->h4,sigmoid(fc)->out

# 生成模型
def generator(z,y,train=True):
    with tf.variable_scope("generator") as scope:
      if not y_dim:
        s_h, s_w = output_height, output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        z_,h0_w, h0_b = linear(
            z, gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

        h0 = tf.reshape(z_, [-1, s_h16, s_w16, gf_dim * 8])
        h0 = tf.nn.relu(norm('g_bn0',h0))

        h1, h1_w, h1_b = deconv2d(
            h0, [batch_size, s_h8, s_w8, gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(norm('g_bn1',h1))

        h2, h2_w, h2_b = deconv2d(
            h1, [batch_size, s_h4, s_w4, gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(norm('g_bn2',h2))

        h3, h3_w, h3_b = deconv2d(
            h2, [batch_size, s_h2, s_w2, gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(norm('g_bn3',h3))

        h4, h4_w, h4_b = deconv2d(
            h3, [batch_size, s_h, s_w, c_dim], name='g_h4', with_w=True)
        return tf.nn.tanh(h4)
      else:
        s_h, s_w = output_height, output_width
        s_h2, s_h4 = int(s_h/2), int(s_h/4)
        s_w2, s_w4 = int(s_w/2), int(s_w/4)

        # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
        yb = tf.reshape(y, [batch_size, 1, 1, y_dim])
        z = tf.concat([z, y], 1)

        h0 = tf.nn.relu(norm('g_bn0',linear(z, gfc_dim, 'g_h0_lin')))
        h0 = tf.concat([h0, y], 1)

        h1 = tf.nn.relu(norm('g_bn1',linear(h0, gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
        h1 = tf.reshape(h1, [batch_size, s_h4, s_w4, gf_dim * 2])

        h1 = conv_cond_concat(h1, yb)

        h2 = tf.nn.relu(norm('g_bn2',deconv2d(h1,
            [batch_size, s_h2, s_w2, gf_dim * 2], name='g_h2')))
        h2 = conv_cond_concat(h2, yb)

        return tf.nn.sigmoid(
            deconv2d(h2, [batch_size, s_h, s_w, c_dim], name='g_h3'))

