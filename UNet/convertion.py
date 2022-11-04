import tensorflow as tf
import datetime
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# decode the data
class DataDecoder():
    def __init__(self, trainset, testset):
        self.trainset = trainset
        self.testset = testset
        print('DataDecoder Initialized:\nEncoded train data: '+self.trainset+'\nEncoded test data: '+self.testset+'\n')

    def read_file(self, filepath):
        f = h5py.File(filepath, "r")
        color_codes, rgb, seg = f['color_codes'][:], f['rgb'][:], f['seg'][:]
        return f, color_codes, rgb, seg

    def decode_train_rgb(self):
        _, _, rgb_train, _ = self.read_file(self.trainset)
        num_train = rgb_train.shape[0]
        for i in range(num_train):
            image = Image.fromarray(rgb_train[i])
            image.save('./Train_rgb/' + str(i) + '_rgb.png')

    def decode_train_seg(self):
        _, _, _, seg_train = self.read_file(self.trainset)
        num_train = seg_train.shape[0]
        for i in range(num_train):
            image = Image.fromarray(np.squeeze((seg_train[i])))     # You have to squeeze it !
            image.save('./Train_seg/' + str(i) + '_seg.png')

    def decode_test_rgb(self):
        _, _, rgb_test, _ = self.read_file(self.testset)
        num_test = rgb_test.shape[0]
        for i in range(num_test):
            image = Image.fromarray(rgb_test[i])
            image.save('./Test_rgb/' + str(i) + '_rgb.png')

    def decode_test_seg(self):
        _, _, _, seg_test = self.read_file(self.testset)
        num_test = seg_test.shape[0]
        for i in range(num_test):
            image = Image.fromarray(np.squeeze(seg_test[i]))
            image.save('./Test_seg/' + str(i) + '_seg.png')

Data = DataDecoder('./lab2_train_data.h5','./lab2_test_data.h5')

Data.read_file('./lab2_train_data.h5')
Data.read_file('./lab2_test_data.h5')

Data.decode_train_rgb()
Data.decode_train_seg()
Data.decode_test_rgb()
Data.decode_test_seg()