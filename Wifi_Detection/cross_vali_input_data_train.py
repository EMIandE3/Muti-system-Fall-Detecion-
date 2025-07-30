from __future__ import print_function
import gzip
import os
import numpy as np,numpy
import csv
import glob
import pandas as pd

class DataSet(object):
    def __init__(self, images,fake_data=False):  # labels
        # assert images.shape[0] == labels.shape[0], (
        #         "images.shape: %s labels.shape: %s" % (images.shape,
        #                                                 labels.shape))
        self._num_examples = images.shape[0]
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
        self._images = images
        # self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
    @property
    def images(self):
        return self._images
    # @property
    # def labels(self):
    #     return self._labels
    @property
    def num_examples(self):
        return self._num_examples
    @property
    def epochs_completed(self):
        return self._epochs_completed
    def next_batch(self, batch_size, fake_data=False):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:  # 只有当把所有的东西取完了才会进行打乱，我一个一个输入不会打乱
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            # self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end]  # , self._labels[start:end]

def csv_import():
    x_dic = {}
    y_dic = {}
    print("csv file importing...")

    for i in ['test']:  # 如果要改类别，这里面加载数据也要改

        xx = np.array(pd.read_csv("./input_files/xx_1000_60_" + str(i) + ".csv", header=None))  # , skiprows = skip_idx
        # yy = np.array(pd.read_csv("./input_files_test/yy_1000_60_" + str(i) + ".csv", header=None))  # , skiprows = skip_idx

        # eliminate the NoActivity Data ,在进行预测的时候是不能删除的，删了看看
        # rows, cols = np.where(yy>0)
        # xx = np.delete(xx, rows[ np.where(cols==0)],0)
        # yy = np.delete(yy, rows[ np.where(cols==0)],0)

        xx = xx.reshape(len(xx),1000,90)

        # 1000 Hz to 500 Hz (To avoid memory error)  隔一行取出一个数据1000*90变成500*90
        xx = xx[:,::2,:90]

        x_dic[str(i)] = xx
        # y_dic[str(i)] = yy

        print(str(i), "finished...", "xx=", xx.shape)  # , "yy=",  yy.shape

    return x_dic["fall"]
        # y_dic["bed"], y_dic["fall"], y_dic["pickup"], y_dic["run"], y_dic["sitdown"], y_dic["standup"], y_dic["walk"]
