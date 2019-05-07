import imgaug
from imgaug import augmenters as iaa

sometimes = lambda aug: iaa.Sometimes(0.5, aug) #建立lambda表达式，

seq = iaa.Sequential(
    [
        iaa.Fliplr(0.5),  # 对50%的图像进行镜像翻转
        sometimes(iaa.Crop(percent=(0, 0.1))), # crop的幅度为0到15%

        sometimes(iaa.Affine(  # 对一部分图像做仿射变换
            scale={"x": (0.85, 1.15), "y": (0.85, 1.15)},  # 图像缩放为80%到120%之间
            translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},  # 平移±20%之间
            rotate=(-20, 20),  # 旋转±25度之间
            shear=(-10, 10),  # 剪切变换±12度，（矩形变平行四边形）
            order=[0, 1],  # 使用最邻近差值或者双线性差值
            cval=(0, 255),  # 全白全黑填充
            mode= imgaug.ALL  # 定义填充图像外区域的方法
        )),

        # 使用下面的0个到5个之间的方法去增强图像。注意SomeOf的用法
        iaa.SomeOf((0, 2),
                   [
                       # 用高斯模糊，均值模糊，中值模糊中的一种增强。注意OneOf的用法
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 1.5)),
                           iaa.AverageBlur(k=(2, 5)),  # 核大小2~7之间，k=((5, 7), (1, 3))时，核高度5~7，宽度1~3
                           iaa.MedianBlur(k=(3, 7)),
                       ]),

                       # 锐化处理
                       iaa.Sharpen(alpha=(0, 1.0), lightness=(0.85, 1.3)),

                       # 加入高斯噪声
                       iaa.AdditiveGaussianNoise(
                           loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                       ),

                       # 将1%到10%的像素设置为黑色
                       iaa.Dropout((0.01, 0.1), per_channel=0.5),

                       # 每个像素随机加减-10到10之间的数
                       iaa.Add((-10, 10), per_channel=0.5),

                       # 将整个图像的对比度变为原来的一半或者二倍
                       iaa.ContrastNormalization((0.7, 1.4), per_channel=0.5),

                       # 把像素移动到周围的地方。这个方法在mnist数据集增强中有见到
                       sometimes(
                           iaa.ElasticTransformation(alpha=(0.5, 2.5), sigma=0.25)
                       ),

                       # 扭曲图像的局部区域
                       sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                   ],

                   random_order=True  # 随机的顺序把这些操作用在图像上
                   )
    ],
    random_order=True  # 随机的顺序把这些操作用在图像上
)

def iaa_data_augment(image):
    image_aug = seq.augment_image(image)
    return(image_aug)
