import tensorflow as tf
import datetime
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import Visualize
import utils
from math import e, log
from scipy.stats import entropy

import h5py
import glob

import tf_slim as slim
from keras.utils.vis_utils import plot_model
from keras.models import load_model

val_img = glob.glob('./Test_rgb/*_rgb.png')  # tf.io.glob.glob
val_label = glob.glob('./Test_seg/*_seg.png')
img_names = []
for path in val_img:
    img_names.append(path.split('/Test_rgb')[1].split('_rgb.png')[0])

val_label = ['./Test_seg/' + name + '_seg.png' for name in img_names]

val_ds = tf.data.Dataset.from_tensor_slices((val_img, val_label))

BATCH_SIZE = 32
auto = tf.data.experimental.AUTOTUNE

val_ds = val_ds.map(utils.load_img_val, num_parallel_calls=auto)

val_ds = val_ds.cache().batch(BATCH_SIZE)

# epistemic uncertainty and aleatoric uncertainty

np.set_printoptions(threshold=np.inf)

def variance(n):
    res = np.empty((256, 256))
    for i in range(255):
        for j in range(255):
            res[i][j] = np.var(n[i, j, :])
    return res

# load the model
model = load_model('./model.h5')

tf.keras.utils.plot_model(model, to_file='Unet_Structure.png', show_shapes=True)

count = 0

# to store μ
exist = os.path.exists('./Examples')
if not exist:
    os.makedirs('./Examples')

# 5 examples
for val_img, label in val_ds.take(5):
    concat_pred_label_p = None

    pred_label_a = model.predict(val_img)
    a_tmp = tf.reduce_mean(pred_label_a,axis=0)

    # plt.imshow(tf.image.resize(val_img[0],[128,256]))
    # plt.savefig('./Examples/original'+str(count)+'.jpg')

    for i in range(10):
        # 10 times
        pred_label = model.predict(val_img)
        pred_label_p = tf.reduce_max(pred_label, axis=-1)
        pred_label_p, _ = tf.nn.moments(pred_label_p, axes=0)
        pred_label = tf.argmax(pred_label, axis=-1)
        pred_label, _ = tf.nn.moments(pred_label, axes=0)
        if i == 0:
            plt.imshow(tf.keras.preprocessing.image.array_to_img(tf.image.resize(tf.expand_dims(pred_label, -1),[128,256])))
            plt.savefig('./Examples/example' + str(count) + '.jpg')
        if concat_pred_label_p is None:
            concat_pred_label_p = tf.expand_dims(pred_label_p, -1)
            continue
        concat_pred_label_p = tf.concat([concat_pred_label_p, tf.expand_dims(pred_label_p, -1)], axis=-1)

    mean, var_ = tf.nn.moments(concat_pred_label_p, axes=[2])
    var = variance(concat_pred_label_p)

    al = entropy(a_tmp, axis=-1)
    print(al.shape)

    Visualize.Visualize(var, 'epistemic_' + str(count))
    Visualize.Visualize(al, 'aleatoric_' + str(count))
    count += 1