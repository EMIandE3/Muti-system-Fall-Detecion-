# 使用预训练的模型对测试集数据进行预测，并打印出预测结果的形状。
import keras
from scipy.io import loadmat
import matplotlib.pyplot as plt
import glob
import numpy as np
import pandas as pd
import math
import os
from keras.layers import *
from keras.models import *
from keras.optimizers import *

Batch_size = 30 
Lens = 300  #验证集行数，1200-300为训练集行数

TEST_MANIFEST_DIR = r'E:\A_Shi_Chuang\Vgg_classfy\test\input_fall_170310_1136_01_test_exp.csv'
def ts_gen(path=TEST_MANIFEST_DIR, batch_size=Batch_size):
    img_list = pd.read_csv(path)
    img_list = np.array(img_list)[:Lens]
    print("Found %s test items." % len(img_list))
    print("list 1 is", img_list[0, -1])
    steps = math.ceil(len(img_list) / batch_size)
    while True:
        for i in range(steps):
            batch_list = img_list[i * batch_size:i * batch_size + batch_size]
            batch_x = np.array([file for file in batch_list[:, 1:]])
            yield batch_x

if __name__ == "__main__":
        test_iter = ts_gen()
        model = load_model("best_model.05-0.0001.h5")
        pres = model.predict_generator(generator=test_iter, steps=math.ceil(Lens / Batch_size), verbose=1)
        print(pres.shape)
        ohpres = np.argmax(pres, axis=1) #取最大概率
        print(ohpres.shape)
