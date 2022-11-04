import matplotlib.pyplot as plt
import numpy as np
import operator
import os
import sys
import tensorflow as tf

# 256*256 映射到 0~1 区间

def Transform(input):
    if not (operator.eq(input.shape, (256, 256))):
        print("not matched")
        sys.exit(1)

    max = np.max(input)
    min = np.min(input)
    for i in range(256):
        for j in range(256):
            input[i][j] = (input[i][j] - min) / (max - min)
    return input


##img:256x256的数组，i的类型为str!!!!
def Visualize(img, i):
    plt.matshow(img, cmap=plt.get_cmap('RdBu'), alpha=0.5)
    exist = os.path.exists('./Uncertainty')
    if not exist:
        os.makedirs('./Uncertainty')
    plt.savefig('./Uncertainty/test_'+i+'.jpg')
    plt.show()

# Test
# test_input = np.random.rand(128, 256)
test_input = np.ones((256,256),dtype=float)
Visualize(test_input, '1')
# Visualize(Transform(test_input), '1')
print(test_input)
