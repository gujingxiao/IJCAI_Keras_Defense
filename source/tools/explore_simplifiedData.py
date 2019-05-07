import argparse
import matplotlib.pyplot as plt
from utils.data_generator import train_generator, valid_generator

def main():
    args = argparser.parse_args()

    shuffle_data_dir = args.shuffle_data_dir
    base_size = args.base_size
    train_size = args.train_size
    splited_folders = args.splited_folders
    number_of_classes = args.number_of_classes
    augmentation = args.augmentation
    batch_size = args.batch_size
    line_width = args.line_width

    train_datagen = train_generator(dp_dir=shuffle_data_dir, size=train_size, batchsize=batch_size, ks=range(splited_folders - 1),
                                    lw=line_width, augment=augmentation, num_classes=number_of_classes, base_size=base_size)

    valid_datagen = valid_generator(dp_dir=shuffle_data_dir, size=train_size, batchsize=batch_size, ks=splited_folders - 1,
                                    lw=line_width, augment=False, num_classes=number_of_classes, base_size=base_size)

    for x in range(5):
        train_image, train_label = next(train_datagen)

        n = 6
        # show train images
        fig, axs = plt.subplots(nrows=n, ncols=n, figsize=(16, 16))
        for i in range(n ** 2):
            ax = axs[i // n, i % n]
            ax.imshow((-train_image[i, :, :] + 1) / 2, cmap=plt.cm.gray)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

        # show valid images
        valid_image, valid_label = next(valid_datagen)
        fig, axs = plt.subplots(nrows=n, ncols=n, figsize=(16, 16))
        for i in range(n ** 2):
            ax = axs[i // n, i % n]
            ax.imshow((-valid_image[i, :, :] + 1) / 2, cmap=plt.cm.gray)
            ax.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--shuffle_data_dir", default="../data/shuffle_csv/")
    # simplified数据都按照256x256给的点，所以base_size是256
    argparser.add_argument("--base_size", default=256, type=int)
    # train_size是实际训练使用的尺寸，会自动缩放
    argparser.add_argument("--train_size", default=128, type=int)
    # shuffle后的csv共有800个，默认最后一个为验证集
    argparser.add_argument("--splited_folders", default=600, type=int)
    # 共有340类
    argparser.add_argument("--number_of_classes", default=340, type=int)
    argparser.add_argument("--batch_size", default=64, type=int)
    # csv数据转成图像时画线的宽度
    argparser.add_argument("--line_width", default=4, type=int)
    # 数据量太大，数据增强可以暂时先不用
    argparser.add_argument("--augmentation", default=False)
    main()