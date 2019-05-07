import os
import argparse
import pandas as pd
from keras.layers import Input, Dense, Dropout, Concatenate
from keras.metrics import categorical_accuracy, categorical_crossentropy
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.mobilenet import MobileNet
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.resnet import ResNet50, ResNet101, ResNet152
from keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.applications.resnext import ResNeXt50, ResNeXt101
from keras.utils import multi_gpu_model
from utils.denseMoE import DenseMoE
from utils.data_generator import valid_generator

def create_model(type, input_image):
    # MobileNet: ImageNet-224x224 Top1-70.424% Model_Size-4.3M
    if type == "mobilenet_v1":
        model = MobileNet(input_tensor=input_image, include_top=False, weights="./pretrained_model/mobilenet_no_top.h5",pooling='avg')
    # Resnet50： ImageNet-224x224 Top1-74.928% Model_Size-25.6M
    elif type == "resnet50":
        model = ResNet50(input_tensor=input_image, include_top=False, weights="./pretrained_model/resnet50_no_top.h5",pooling='avg')
    # Resnet101： ImageNet-224x224 Top1-76.420% Model_Size-44.7M
    elif type == "resnet101":
        model = ResNet101(input_tensor=input_image, include_top=False, weights="./pretrained_model/resnet101_no_top.h5",pooling='avg')
    # Resnet152: ImageNet-224x224 Top1-76.604% Model_Size-60.4M
    elif type == "resnet152":
        model = ResNet152(input_tensor=input_image, include_top=False, weights="./pretrained_model/resnet152_no_top.h5",pooling='avg')
    # Resnet50V2： ImageNet-224x224 Top1-75.960% Model_Size-25.6M
    elif type == "resnet50v2":
        model = ResNet50V2(input_tensor=input_image, include_top=False, weights="./pretrained_model/resnet50v2_no_top.h5",pooling='avg')
    # Resnet101V2： ImageNet-224x224 Top1-77.234% Model_Size-44.7M
    elif type == "resnet101v2":
        model = ResNet101V2(input_tensor=input_image, include_top=False, weights="./pretrained_model/resnet101v2_no_top.h5",pooling='avg')
    # Resnet152V2: ImageNet-224x224 Top1-78.032% Model_Size-60.4M
    elif type == "resnet152v2":
        model = ResNet152V2(input_tensor=input_image, include_top=False, weights="./pretrained_model/resnet152_no_top.h5",pooling='avg')
    # ResneXt50： ImageNet-224x224 Top1-77.740% Model_Size-25.1M
    elif type == "resnext50":
        model = ResNeXt50(input_tensor=input_image, include_top=False, weights="./pretrained_model/resnext50_no_top.h5",pooling='avg')
    # ResneXt101： ImageNet-224x224 Top1-78.730% Model_Size-44.3M
    elif type == "resnext101":
        model = ResNeXt101(input_tensor=input_image, include_top=False, weights="./pretrained_model/resnext101_no_top.h5",pooling='avg')
    # Densenet121: ImageNet-224x224 Top1-74.972% Model_Size-8.1M
    elif type == "densenet121":
        model = DenseNet121(input_tensor=input_image, include_top=False, weights="./pretrained_model/densenet121_no_top.h5",pooling='avg')
    # Densenet169: ImageNet-224x224 Top1-76.176% Model_Size-14.3M
    elif type == "densenet169":
        model = DenseNet169(input_tensor=input_image, include_top=False, weights="./pretrained_model/densenet169_no_top.h5",pooling='avg')
    # Densenet201: ImageNet-224x224 Top1-77.320% Model_Size-20.2M
    elif type == "densenet201":
        model = DenseNet201(input_tensor=input_image, include_top=False, weights="./pretrained_model/densenet201_no_top.h5",pooling='avg')
    # Xception: ImageNet-299x299 Top1-79.006% Model_Size-22.9M
    elif type == "xception":
        model = Xception(input_tensor=input_image, include_top=False, weights="./pretrained_model/xception_no_top.h5",pooling='avg')
    # InceptionV3: ImageNet-299x299 Top1-77.898% Model_Size-23.9M
    elif type == "inceptionv3":
        model = InceptionV3(input_tensor=input_image, include_top=False, weights="./pretrained_model/inceptionv3_no_top.h5",pooling='avg')
    # InceptionresnetV2: ImageNet-299x299 Top1-80.256% Model_Size-55.9M
    elif type == "inceptionresnetV2":
        model = InceptionResNetV2(input_tensor=input_image, include_top=False, weights="./pretrained_model/inception_resnet_v2_no_top.h5",pooling='avg')
    else:
        raise Exception("Unsupported model type: '{}".format(type))
    return model

def main():
    args = argparser.parse_args()

    input_dir = args.input_dir
    val_dir = args.val_list
    train_size = args.train_size
    number_of_classes = args.number_of_classes
    model_type = args.model_type
    batch_size = args.batch_size
    lr_max = args.lr_max
    finetune_dir = args.finetune_dir
    multi_gpus = args.multi_gpus
    use_MoE = args.use_MoE

    # Build Model
    input_image = Input(shape=(train_size, train_size, 3))
    base_model_densenet121 = create_model("densenet121", input_image)
    base_model_xception = create_model("xception", input_image)

    for layer in base_model_densenet121.layers:
        print(layer.name, layer.output_shape)
        layer.trainable = False

    for layer in base_model_xception.layers:
        print(layer.name, layer.output_shape)
        layer.trainable = False

    ensemble_concat = Concatenate(axis=-1)([base_model_densenet121.output, base_model_xception.output])
    predict = Dense(number_of_classes, activation='softmax')(ensemble_concat)
    model = Model(inputs=input_image, outputs=predict)

    if multi_gpus == True:
        paralleled_model = multi_gpu_model(model=model, gpus=2)
        if finetune_dir is not None:
            print('Finetune from a IJCAI pretrained model: ', finetune_dir)
            paralleled_model.load_weights(finetune_dir + ".h5")
            model.save_weights(finetune_dir + "_single.h5")
        else:
            print('No finetuning. Start from a ImageNet pretrained model.')
        paralleled_model.compile(optimizer=Adam(lr=lr_max), loss='categorical_crossentropy',
                                 metrics=[categorical_crossentropy, categorical_accuracy])
    else:
        if finetune_dir is not None:
            print('Finetune from a IJCAI pretrained model: ', finetune_dir)
            model.load_weights(finetune_dir + "_single.h5")
        else:
            print('No finetuning. Start from a ImageNet pretrained model.')
        model.compile(optimizer=Adam(lr=lr_max), loss='categorical_crossentropy',
                  metrics=[categorical_crossentropy, categorical_accuracy])

    # Evaluation
    val_list = pd.read_csv(os.path.join(input_dir, val_dir))
    length = 5
    losses = 0.0
    scores = 0.0
    print("Start Evalutation: ")
    for index in range(length):
        if index == 0:
            augmentation = False
        else:
            augmentation = True
        val_gen = valid_generator(val_list, train_size, batch_size, augmentation, number_of_classes)

        if multi_gpus == True:
            preds = paralleled_model.evaluate_generator(val_gen, steps=int(len(val_list) / batch_size), verbose=0)
        else:
            preds = model.evaluate_generator(val_gen, steps=int(len(val_list) / batch_size), verbose=0)

        print("Num: {}, Augmentation: {},  Loss: {}, Scores: {}".format(index, augmentation, preds[0], preds[2]))
        if index != 0:
            losses += preds[0]
            scores += preds[2]
    print("Average Augmentation Loss: {}, Average Augmentation Score: {}".format(losses / (length - 1), scores / (length - 1)))


"""
README

#1 安装必需的计算库
   Python3.5（千万不要用2.7, 千万不要用2.7, 千万不要用2.7）
   CUDA9.0 + Cudnn7.0（比CUDA8.0 + Cudnn快了近一倍）
   Tensorflow-gpu==1.12.0
   Keras==2.2.2
   opencv-python==3.4.2.16
   其他的包根据程序补充即可

#2 工程结构
   models:                         存放训练好的模型
   pretrained_model:               存放imagenet下的预训练模型
   utils:  - classfication_models  第三方的分类模型脚本
           - denseMoE.py           采用MOE作为分类器训练
           - preprocessing.py      数据预处理、数据增强
           - dataGenerator.py      数据读取与载入
   train.py                        训练脚本
   defense.py                      生成预测脚本
"""

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    # 所有的相对路径建议不要修改，按照名称创建即可
    argparser.add_argument("--input_dir", default="../data/")
    argparser.add_argument("--val_list", default="IJCAI_2019_AAAC_val.csv")
    # 是否使用多GPU
    argparser.add_argument("--multi_gpus", default=True)
    # finetune_dir可以加载使用训练好的模型，直接改路径即可
    argparser.add_argument("--finetune_dir", default="./models/ijcai_ensemble")
    # train_size是实际训练使用的尺寸，会自动缩放
    argparser.add_argument("--train_size", default=299, type=int)
    # 共有110类
    argparser.add_argument("--number_of_classes", default=110, type=int)
    # 是否使用Mixture of Experts处理分类
    argparser.add_argument("--use_MoE", default=False)
    argparser.add_argument("--batch_size", default=64, type=int)
    # 根据经验一般都用(0.00001,0.001)这个范围
    argparser.add_argument("--lr_max", default=0.001, type=float)
    # 使用哪个网络结构
    argparser.add_argument("--model_type", default="ensemble")
    main()