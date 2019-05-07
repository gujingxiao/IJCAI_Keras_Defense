from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Concatenate
from keras.metrics import categorical_accuracy
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet121

tf.flags.DEFINE_string(
    'checkpoint_path', './models/ijcai_ensemble_single.h5', 'Path to checkpoint for inception network.')
tf.flags.DEFINE_string(
    'input_dir', '../data/dev/dev_data/', 'Input directory with images.')
tf.flags.DEFINE_string(
    'output_file', './defence.csv', 'Output file to save labels.')
tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')
tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')
tf.flags.DEFINE_integer(
    'batch_size', 16, 'Batch size to processing images')
tf.flags.DEFINE_integer(
    'num_classes', 110, 'How many classes of the data set')
FLAGS = tf.flags.FLAGS

def main():
    input_image = Input(shape=(FLAGS.image_height, FLAGS.image_width, 3))

    base_model_densenet121 = DenseNet121(input_tensor=input_image, include_top=False, weights=None, pooling='avg')
    base_model_xception = Xception(input_tensor=input_image, include_top=False, weights=None, pooling='avg')

    ensemble_concat = Concatenate(axis=-1)([base_model_densenet121.output, base_model_xception.output])
    predict = Dense(FLAGS.num_classes, activation='softmax')(ensemble_concat)
    model = Model(inputs=input_image, outputs=predict)


    # base_model = Xception(input_tensor=input_image, include_top=False, weights=None, pooling='avg')
    # predict = Dense(FLAGS.num_classes)(base_model.output)
    # model = Model(inputs=input_image, outputs=predict)
    model.load_weights(FLAGS.checkpoint_path)
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy',
              metrics=[categorical_accuracy])

    with open(FLAGS.output_file, 'w') as out_file:
        for filepath in tf.gfile.Glob(os.path.join(FLAGS.input_dir, '*.png')):
            image = cv2.resize(cv2.imread(filepath), (FLAGS.image_height, FLAGS.image_width))
            image = np.array(image)
            image = image / 255.0
            image = np.expand_dims(image, axis=0)

            labels = model.predict(image)
            newlabels = np.argsort(-labels, axis=1)[0, 0]
            filename = os.path.basename(filepath)
            out_file.write('{0},{1}\n'.format(filename, newlabels))

if __name__ == '__main__':
    main()