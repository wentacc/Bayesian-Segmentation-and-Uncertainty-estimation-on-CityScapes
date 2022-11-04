import tensorflow as tf
import datetime
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob

import tf_slim as slim
import Visualize
import utils

images = glob.glob('./Train_rgb/*_rgb.png')  # tf.io.glob.glob
labels = glob.glob('./Train_seg/*_seg.png')

image_names = []
for paths in images:
    image_names.append(paths.split('/Train_rgb')[1].split('_rgb.png')[0])

labels = ['./Train_seg/' + name + '_seg.png' for name in image_names]

index = np.random.permutation(2975)
images = np.array(images)[index]
labels = np.array(labels)[index]

val_img = glob.glob('./Test_rgb/*_rgb.png')  # tf.io.glob.glob
val_label = glob.glob('./Test_seg/*_seg.png')

image_names = []
for paths in val_img:
    image_names.append(paths.split('/Test_rgb')[1].split('_rgb.png')[0])

val_label = ['./Test_seg/' + name + '_seg.png' for name in image_names]

train_data = tf.data.Dataset.from_tensor_slices((images, labels))
val_data = tf.data.Dataset.from_tensor_slices((val_img, val_label))

BATCH_SIZE = 32
BUFFER_SIZE = 300
STEPS_PER_EPOCH = 2975 // BATCH_SIZE
VALIDATION_STEPS = 500 // BATCH_SIZE
auto = tf.data.experimental.AUTOTUNE

train_data = train_data.map(utils.load_img_train, num_parallel_calls=auto)
val_data = val_data.map(utils.load_img_val, num_parallel_calls=auto)

train_data = train_data.cache().repeat().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(auto)
val_data = val_data.cache().batch(BATCH_SIZE)


# Unet Model with dropout

def create_model():
    inputs = tf.keras.layers.Input(shape=(256, 256, 3))

    layer0 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    layer0 = tf.keras.layers.BatchNormalization()(layer0)
    layer0 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(layer0)
    layer0 = tf.keras.layers.BatchNormalization()(layer0)  # 256*256*64

    layer1 = tf.keras.layers.MaxPooling2D(padding='same')(layer0)  # 128*128*64

    layer1 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(layer1)
    layer1 = tf.keras.layers.BatchNormalization()(layer1)
    layer1 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(layer1)
    layer1 = tf.keras.layers.BatchNormalization()(layer1)  # 128*128*128

    layer2 = tf.keras.layers.MaxPooling2D(padding='same')(layer1)  # 64*64*128

    layer2 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(layer2)
    layer2 = tf.keras.layers.BatchNormalization()(layer2)
    layer2 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(layer2)
    layer2 = tf.keras.layers.BatchNormalization()(layer2)  # 64*64*256

    layer3 = tf.keras.layers.MaxPooling2D(padding='same')(layer2)  # 32*32*256

    layer3 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(layer3)
    layer3 = tf.keras.layers.BatchNormalization()(layer3)
    layer3 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(layer3)
    layer3 = tf.keras.layers.BatchNormalization()(layer3)  # 32*32*512

    layer4 = tf.keras.layers.MaxPooling2D(padding='same')(layer3)  # 16*16*512

    layer4 = tf.keras.layers.Conv2D(1024, 3, padding='same', activation='relu')(layer4)
    layer4 = tf.keras.layers.BatchNormalization()(layer4)
    layer4 = tf.keras.layers.Conv2D(1024, 3, padding='same', activation='relu')(layer4)
    layer4 = tf.keras.layers.BatchNormalization()(layer4)  # 16*16*1024

    layer5 = tf.keras.layers.Conv2DTranspose(512, 2, strides=2, padding='same', activation='relu')(layer4)
    layer5 = tf.keras.layers.BatchNormalization()(layer5)  # 32*32*512

    layer6 = tf.concat([layer3, layer5], axis=-1)  # 32*32*1024

    layer6 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(layer6)
    layer6 = tf.keras.layers.BatchNormalization()(layer6)
    layer6 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(layer6)
    layer6 = tf.keras.layers.BatchNormalization()(layer6)  # 32*32*512

    layer7 = tf.keras.layers.Conv2DTranspose(256, 2, strides=2, padding='same', activation='relu')(layer6)
    layer7 = tf.keras.layers.BatchNormalization()(layer7)  # 64*64*256

    layer8 = tf.concat([layer2, layer7], axis=-1)  # 64*64*512

    layer8 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(layer8)
    layer8 = tf.keras.layers.BatchNormalization()(layer8)
    layer8 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(layer8)
    layer8 = tf.keras.layers.BatchNormalization()(layer8)  # 64*64*256

    layer9 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding='same', activation='relu')(layer8)
    layer9 = tf.keras.layers.BatchNormalization()(layer9)  # 128*128*128

    layer10 = tf.concat([layer1, layer9], axis=-1)  # 128*128*256
    layer10 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(layer10)
    layer10 = tf.keras.layers.BatchNormalization()(layer10)
    layer10 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(layer10)
    layer10 = tf.keras.layers.BatchNormalization()(layer10)  # 128*128*128
    layer10 = tf.keras.layers.Dropout(0.5, noise_shape=None)(layer10)

    layer11 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding='same', activation='relu')(layer10)
    layer11 = tf.keras.layers.BatchNormalization()(layer11)  # 256*256*64

    layer12 = tf.concat([layer0, layer11], axis=-1)  # 256*256*128

    layer12 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(layer12)
    layer12 = tf.keras.layers.BatchNormalization()(layer12)
    layer12 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(layer12)
    layer12 = tf.keras.layers.BatchNormalization()(layer12)  # 256*256*64

    outputs = tf.keras.layers.Conv2D(34, 1, activation='softmax')(layer12)  # 256*256*34

    return tf.keras.Model(inputs=inputs, outputs=outputs)


# create the model
model = create_model()
model.summary()


class MeanIoU(tf.keras.metrics.MeanIoU):
    def __call__(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().__call__(y_true, y_pred, sample_weight=sample_weight)


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

history = model.fit(train_data, steps_per_epoch=STEPS_PER_EPOCH, validation_data=val_data,
                    validation_steps=VALIDATION_STEPS, epochs=10)

# save the model
model.save('./model.h5')
