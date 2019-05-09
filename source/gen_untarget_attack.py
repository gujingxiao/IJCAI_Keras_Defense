import os
import cv2
import numpy as np
import argparse
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense
from keras.applications.mobilenet import MobileNet
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.resnet import ResNet50, ResNet101, ResNet152
from keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.applications.resnext import ResNeXt50, ResNeXt101
import keras.backend as K

def preds2catids(predictions, topk=1):
    return np.argsort(-predictions, axis=1)[:, :topk]

def compute_perturbation(image, adv_image):
    return np.linalg.norm(adv_image - image)


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

    save_dir = args.save_dir
    gen_num = args.gen_num
    input_dir = args.input_dir
    val_dir = args.val_list
    train_size = args.train_size
    number_of_classes = args.number_of_classes
    model_type = args.model_type
    model_dir = args.model_dir

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_image_list = []
    save_label_list = []
    # Build Model
    input_image = Input(shape=(train_size, train_size, 3))

    base_model = create_model(model_type, input_image)
    predict = Dense(number_of_classes, activation='softmax')(base_model.output)
    model = Model(inputs=input_image, outputs=predict)
    print('Finetune from a IJCAI pretrained model: ', model_dir)
    model.load_weights(model_dir + "_single.h5")


    # Untartget Attack
    val_list = pd.read_csv(os.path.join(input_dir, val_dir))
    # Get current session (assuming tf backend)
    sess = K.get_session()
    # Initialize adversarial example with input image
    all_batches_index = np.arange(0, len(val_list))
    image_dir = np.array(val_list['image_dir'])
    gt_label = np.array(val_list['label'])
    # Random shuffle indexes every epoch
    np.random.shuffle(all_batches_index)
    count = 0
    for index in all_batches_index:
        if os.path.exists(os.path.join('../', image_dir[index])):
            count += 1
            image = cv2.resize(cv2.imread(os.path.join('../', image_dir[index])), (train_size, train_size))
            image = np.array(image)
            image = image / 255.0
            image = np.expand_dims(image, axis=0)
            gt_class = gt_label[index]
            initial_preds = model.predict(image)
            initial_class = np.argmax(initial_preds)

            x_adv = image
            # Added noise
            x_noise = np.zeros_like(image)
            # One hot encode the initial class
            target = K.one_hot(initial_class, args.number_of_classes)
            # Set variables
            epsilon = np.random.randint(5, 100) / 10000.0
            iters = np.random.randint(2, 7)

            for i in range(iters):
                # Get the loss and gradient of the loss wrt the inputs
                loss = K.categorical_crossentropy(target, model.output)
                grads = K.gradients(loss, model.input)

                # Get the sign of the gradient
                delta = K.sign(grads[0])
                x_noise = x_noise + delta

                # Perturb the image
                x_adv = x_adv + epsilon * delta

                # Get the new image and predictions
                x_adv = sess.run(x_adv, feed_dict={model.input: image})

            preds = model.predict(x_adv)
            predict_class = np.argmax(preds)
            perturbation = compute_perturbation(image[0], x_adv[0])
            print("Count: {}, Index: {}, GT: {}, Ori_Preds: {}, Adv_Preds: {}, Ori_Score: {},  Adv_Score: {}, Perturbation: {}, Adv_iters: {}, Epsilon: {}".format(
                count, index, gt_class, initial_class, predict_class, round(initial_preds[0][initial_class], 5), round(preds[0][initial_class], 5), round(perturbation, 5), iters, epsilon))

            if gt_class != predict_class:
                save_image_list.append(image_dir[index])
                save_label_list.append(gt_class)
                cv2.imwrite(os.path.join(save_dir, image_dir[index].split('/')[-1]), x_adv[0] * 255.0)
            if len(save_image_list) >= gen_num:
                print("Done process {} images.".format(gen_num))
                return
            # cv2.imshow("ori_image", image[0])
            # cv2.imshow("x_adv", x_adv[0])
            # cv2.imshow("dif", (x_adv[0] - image[0]) * 255.0)
            # cv2.waitKey(30)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    # 所有的相对路径建议不要修改，按照名称创建即可
    argparser.add_argument("--input_dir", default="../data/")
    argparser.add_argument("--save_dir", default="../data_adv/")
    argparser.add_argument("--val_list", default="IJCAI_2019_AAAC_train.csv")

    argparser.add_argument("--model_type", default="xception")
    argparser.add_argument("--model_dir", default="./models/ijcai_xception")
    # train_size是实际训练使用的尺寸，会自动缩放
    argparser.add_argument("--train_size", default=299, type=int)
    # Adv generator number
    argparser.add_argument("--gen_num", default=100, type=int)
    # 共有110类
    argparser.add_argument("--number_of_classes", default=110, type=int)

    main()



