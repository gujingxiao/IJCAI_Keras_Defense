import os
import keras
import pandas as pd
import numpy as np
import cv2
from utils.preprocessing import data_augment
from utils.iaa_process import iaa_data_augment

# Train Generator
def train_generator(train_list, size, batchsize, augment, num_classes):
    # Arrange all indexes
    all_batches_index = np.arange(0, len(train_list))
    out_images = []
    out_masks = []
    image_dir = np.array(train_list['image_dir'])
    label_dir = np.array(train_list['label'])
    while True:
        # Random shuffle indexes every epoch
        np.random.shuffle(all_batches_index)
        for index in all_batches_index:
            if os.path.exists(os.path.join('../', image_dir[index])):
                image = cv2.resize(cv2.imread(os.path.join('../', image_dir[index])), (size, size))
                if augment != False:
                    image_aug = iaa_data_augment(image)
                    label = int(label_dir[index])
                    image_aug = np.array(image_aug)
                    image_aug = image_aug / 255.
                    out_images.append(image_aug)
                else:
                    image = np.array(image)
                    image = image / 255.
                    label = int(label_dir[index])
                    out_images.append(image)

                out_masks.append(label)
                if len(out_images) >= batchsize:
                    out_images = np.array(out_images)
                    out_masks = np.array(out_masks)
                    out_masks = keras.utils.to_categorical(out_masks, num_classes=num_classes)
                    yield out_images, out_masks
                    out_images, out_masks = [], []
            else:
                print(image_dir[index], 'does not exist.')


def valid_generator(val_list, size, batchsize, augment, num_classes):
    # Arrange all indexes
    all_batches_index = np.arange(0, len(val_list))
    out_images = []
    out_masks = []
    image_dir = np.array(val_list['image_dir'])
    label_dir = np.array(val_list['label'])
    while True:
        # Random shuffle indexes every epoch
        np.random.shuffle(all_batches_index)
        for index in all_batches_index:
            if os.path.exists(os.path.join('../', image_dir[index])):
                image = cv2.resize(cv2.imread(os.path.join('../', image_dir[index])), (size, size))
                if augment != False:
                    image_aug = iaa_data_augment(image)
                    label = int(label_dir[index])
                    image_aug = np.array(image_aug)
                    image_aug = image_aug / 255.
                    out_images.append(image_aug)
                else:
                    image = np.array(image)
                    image = image / 255.
                    label = int(label_dir[index])
                    out_images.append(image)

                out_masks.append(label)
                if len(out_images) >= batchsize:
                    out_images = np.array(out_images)
                    out_masks = np.array(out_masks)
                    out_masks = keras.utils.to_categorical(out_masks, num_classes=num_classes)
                    yield out_images, out_masks
                    out_images, out_masks = [], []
            else:
                print(image_dir[index], 'does not exist.')