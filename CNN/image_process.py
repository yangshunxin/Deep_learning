''''
    图像处理的Python库：OpenCV、PIL、matplotlib、tensorflow等
'''
import  numpy as np
import matplotlib.pyplot as plt
import  tensorflow as tf


# 打印numpy的数组对象的时候， 中间不省略
np.set_printoptions(threshold=np.inf)

def show_image_tensor(image_tensor):
    # 要求：使用交互式会话
    # 获取图像tensor对象  对应的image对象，image对象是一个[h, w, c]
    # print(image_tensor)
    image = image_tensor.eval()
    # print(image)
    print('图像大小为：{}'.format(image.shape))
    if len(image.shape) == 3 and image.shape[2] == 1:
        # 黑白图像
        plt.imshow(image[:, :, 0], cmap='Greys_r')
        plt.show()
    elif len(image.shape) == 3:
        # 彩色图像
        plt.imshow(image)
        plt.show()

# 1. 交互式会话开启
sess = tf.InteractiveSession()

image_path = 'image/xiaoren.png'
# image_path = 'image/gray.png'
# image_path = 'image/black_white.png'

# 一.图像格式转换
# 读取数据
file_contents = tf.read_file(image_path)
# 将图像数据转换为像素点的数据格式，返回对象为：[height, width, num_channels], 如果是gif的图像返回[num_frames, height, width, num_channels]
# height: 图片的竖直高度的像素大小，
# width： 图片水平宽度的像素大小
# num_channels：图像的通道数，也就是API中的channels的值
# num_frames: 因为gif的图像是一个动态图像，可以将每一个动的画面看成一个静态图像，num_frames相当于在这个gif图像中有多少个静态图像
# 参数channels： 可选值： 0 1 3 4， 默认为0， 一般使用0，1，3， 不建议使用4
# 0： 使用图像的默认通道，也即是图像是几通道的就使用几通道
# 1： 使用灰度级别的图像数据作为返回值（只有一个通道：黑白）
# 3： 使用RGB三通道读取数据
# 4：
image_tensor = tf.image.decode_png(contents=file_contents, channels=3)
# show_image_tensor(image_tensor)

# 二、 图像大小设置 --图片缩放
'''
method取值四个：
BILINEAR = 0  线性差值，默认
NEAREST_NEIGHBOR= 1  最近邻插值，失真最小
BICUBIC = 2 三次差值
AREA = 3 面积差值
'''
# images： 给定需要进行大小转换的图像对应的tensor对象，格式为：[height, width, num_channels] 或者 [batch, height, width, num_channels]
# API返回值核images格式一样，唯一区别是height和width变化为给定的值
resize_image_tensor = tf.image.resize_images(images=image_tensor, size=(200, 200),
                                            method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# show_image_tensor(resize_image_tensor)

# 图片的剪切&填充
# 图片设置大小，通过图片的剪切或者填充（从中间开始计算新图片的大小）
crop_or_pad_image_tensor = tf.image.resize_image_with_crop_or_pad(image_tensor, 200, 200)
# show_image_tensor(crop_or_pad_image_tensor)

# 中间等比例剪切
central_crop_image_tensor = tf.image.central_crop(image_tensor, central_fraction=0.2)
# show_image_tensor(central_crop_image_tensor)

# 填充数据（给定位置开始填充）
pad_to_bounding_box_iamge_tensor = tf.image.pad_to_bounding_box(image_tensor, offset_height=400, offset_width=490,
                                                                target_height=1000, target_width=1000)
# show_image_tensor(pad_to_bounding_box_iamge_tensor)

# 剪切数据（给定位置开始剪切）
crop_to_bounding_box_image_tensor = tf.image.crop_to_bounding_box(image_tensor, offset_height=10, offset_width=40,
                                                                  target_height=200, target_width=300)
# show_image_tensor(crop_to_bounding_box_image_tensor)

# 四、 旋转
# 上下交换
flip_up_down_image_tensor = tf.image.flip_up_down(image_tensor)
# show_image_tensor(flip_up_down_image_tensor)

# 左右交换
flip_left_right_image_tensor = tf.image.flip_left_right(image_tensor)
# show_image_tensor(flip_left_right_image_tensor)

# 转置
transpose_image_tensor = tf.image.transpose_image(image_tensor)
# show_image_tensor(transpose_image_tensor)

# 旋转（90度、180度、270度、、、、）
# k*90度旋转，逆时针旋转、、、
k_rot90_image_tensor = tf.image.rot90(image_tensor, k=3)
# show_image_tensor(k_rot90_image_tensor)

# 五、颜色空间的转换（rgb、hsv、gray）
# 颜色空间的转换必须将image的值转换为float32类型，不能使用uint8类型
float32_image_tensor = tf.image.convert_image_dtype(image_tensor, dtype=tf.float32)
# show_image_tensor(float32_image_tensor)

# rgb -> hsv (h：图像的色彩/色度， s：图像的饱和度， v：图像的亮度)
hsv_image_tensor = tf.image.rgb_to_hsv(float32_image_tensor)
# show_image_tensor(hsv_image_tensor) # 图像看不清

# hsv -> rgb
rgb_iamge_tensor = tf.image.hsv_to_rgb(hsv_image_tensor)
# show_image_tensor(rgb_iamge_tensor) # 图像还原

# rgb -> gray
gray_image_tensor = tf.image.rgb_to_grayscale(rgb_iamge_tensor)
# show_image_tensor(gray_image_tensor)

# gray -> rgb
grayscale_to_rgb_image_tensor = tf.image.grayscale_to_rgb(gray_image_tensor)
# show_image_tensor(grayscale_to_rgb_image_tensor) # 输出还是 黑白的

# 可以从颜色空间中提取图像的轮廓信息（图像的二值化）
a = gray_image_tensor
b = tf.less_equal(a, 0.9)
# 0是黑， 1是白
c = tf.where(condition=b, x=a, y = a-a)
d = tf.where(condition=b, x = c - c + 1, y = c)
# show_image_tensor(d)

# 六、 图像的调整
# 亮度调整
# image：RGB图像信息，设置为float类型和uint8类型不一样，一般建议设置为float类型
# delta：取值范围（-1， 1）之间的float类型的值，表示对于亮度的减弱或者增强的系数值
# 底层执行：rgb -> hsv -> h, s, v*delta -> rgb
adjust_brightness_image_tensor = tf.image.adjust_brightness(image=float32_image_tensor, delta=0.8)
# show_image_tensor(adjust_brightness_image_tensor)

# 色度调整
# image：RGB图像信息，设置为float类型和uint8类型的效果不一样，一般建议设置为float类型
# delta: 取值范围(-1,1）之间的float类型的值，表示对于色调的减弱或者增强的系数值
# 底层执行：rgb -> hsv -> h*delta,s,v -> rgb
adjust_hue_image_tensor = tf.image.adjust_hue(image_tensor, delta=-0.8)
# show_image_tensor(adjust_hue_image_tensor)

# 饱和度调整
# image: RGB图像信息，设置为float类型和unit8类型的效果不一样，一般建议设置为float类型
# saturation_factor: 一个float类型的值，表示对于饱和度的减弱或者增强的系数值，饱和因子
# 底层执行：rgb -> hsv -> h,s*saturation_factor,v -> rgb
adjust_saturation_image_tensor = tf.image.adjust_saturation(image_tensor, saturation_factor=20)
# show_image_tensor(adjust_saturation_image_tensor)

# 对比度调整，公式：(x-mean) * contrast_factor + mean
adjust_contrast_image_tensor = tf.image.adjust_contrast(image_tensor, contrast_factor=10)
# show_image_tensor(adjust_contrast_image_tensor)

# 图像的gamma校正
# image: 要求必须是float类型的数据
# gamma：任意值，Oup = In * Gamma
adjust_gamma_image_tensor = tf.image.adjust_gamma(float32_image_tensor, gamma=10)
# show_image_tensor(adjust_gamma_image_tensor)

# 图像归一化（x - mean）/adjusted_sttdev, adjusted_sttdev=max(stddev, 1.0/sqrt(iamge.NumElements()))
per_image_standardization_iamge_tensor = tf.image.per_image_standardization(image_tensor)
# show_image_tensor(per_image_standardization_iamge_tensor)

# 七、 噪音数据的输入
noisy_image_tensor = image_tensor + tf.cast(5*tf.random_normal(shape=[600, 510, 3], mean=0, stddev=0.1), tf.uint8)
show_image_tensor(noisy_image_tensor)
