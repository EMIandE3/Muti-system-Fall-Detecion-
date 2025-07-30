# 使用卷积神经网络（CNN）对信号数据进行分类，并使用Keras框架进行模型训练和评估。
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

#这里是我导入的训练集数据训练集，大家对应自己的信号数据就好，数据我下面会发，大家可以看一下数据的格式；
MANIFEST_DIR = r'E:\A_Shi_Chuang\Vgg_classfy\train\input_fall_170310_1136_01_train_exp.csv'
Batch_size = 30 
Long = 1200 #csv文件的行数
Lens = 300  #验证集行数，1200-300为训练集行数
def convert2oneHot(index, lens):
    hot = np.zeros((lens,))
    hot[int(index)] = 1
    return(hot)


def xs_gen(path=MANIFEST_DIR, batch_size=Batch_size, train=True, Lens=Lens):
    img_list = pd.read_csv(path)
    if train:
        img_list = np.array(img_list)[:Lens]
        print("Found %s train items." % len(img_list))
        print("list 1 is", img_list[0, -1])
        steps = math.ceil(len(img_list) / batch_size)
    else:
        img_list = np.array(img_list)[Lens:]
        print("Found %s test items." % len(img_list))
        print("list 1 is", img_list[0, -1])
        steps = math.ceil(len(img_list) / batch_size)
    while True:
        for i in range(steps):
            batch_list = img_list[i * batch_size: i * batch_size + batch_size]
            np.random.shuffle(batch_list)
            batch_x = np.array([file for file in batch_list[:, 1:-1]])
            batch_y = np.array([convert2oneHot(label, 2) for label in batch_list[:, -1]])  #注意此处的2要与num_classes一致
            yield batch_x, batch_y


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


TIME_PERIODS = 180 #csv文件的列数



def build_model(input_shape=(TIME_PERIODS,), num_classes=2):
    model = Sequential()
    model.add(Reshape((TIME_PERIODS, 1), input_shape=input_shape))

    model.add(Conv1D(16, 8, strides=2, activation='relu', input_shape=(TIME_PERIODS, 1)))
    model.add(Conv1D(16, 8, strides=2, activation='relu', padding="same"))
    model.add(MaxPooling1D(2)) 

    model.add(Conv1D(64, 4, strides=2, activation='relu', padding="same"))
    model.add(Conv1D(64, 4, strides=2, activation='relu', padding="same"))
    model.add(MaxPooling1D(2))

#     model.add(Conv1D(256, 4, strides=2, activation='relu', padding="same"))
#     model.add(Conv1D(256, 4, strides=2, activation='relu', padding="same"))
#     model.add(MaxPooling1D(2))

#     model.add(Conv1D(512, 2, strides=1, activation='relu', padding="same"))
#     model.add(Conv1D(512, 2, strides=1, activation='relu', padding="same"))
#     model.add(MaxPooling1D(2))

    """ model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))"""
    
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    
    return(model)


Train = True


if __name__ == "__main__":
    if Train == True:
        train_iter = xs_gen()
        val_iter = xs_gen(train=False)

        ckpt = keras.callbacks.ModelCheckpoint(
            filepath='best_model.{epoch:02d}-{val_loss:.4f}.h5',
            monitor='val_loss', save_best_only=True, verbose=1
        )

        model = build_model()
        opt = Adam(0.0002)
        model.compile(loss='categorical_crossentropy',
                      optimizer = opt, metrics=['accuracy'])
        print(model.summary())

        train_history = model.fit_generator(
            generator=train_iter,
            steps_per_epoch=Lens // Batch_size,
            epochs=5,
            initial_epoch=0,
            validation_data=val_iter,
            validation_steps=(Long - Lens) // Batch_size,
            callbacks=[ckpt],
        )

        model.save("finishModel.h5")
    else:
        test_iter = ts_gen()
        model = load_model("best_model.49-0.00.h5")
        pres = model.predict_generator(generator=test_iter, steps=math.ceil(Lens / Batch_size), verbose=1)
        print(pres.shape)
        ohpres = np.argmax(pres, axis=1)
        print(ohpres.shape)
        print(ohpres)


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.ylabel('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

show_train_history(train_history, 'accuracy', 'val_accuracy')

show_train_history(train_history, 'loss', 'val_loss')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(6, 4))
plt.plot(train_history.history['accuracy'], "g--", label="训练集准确率")
plt.plot(train_history.history['val_accuracy'], "g", label="验证集准确率")
plt.plot(train_history.history['loss'], "r--", label="训练集损失函数")
plt.plot(train_history.history['val_loss'], "r", label="验证集损失函数")
plt.title('模型的准确率和损失函数', fontsize=14)
plt.ylabel('准确率和损失函数', fontsize=12)
plt.xlabel('世代数', fontsize=12)
plt.ylim(0)
plt.legend()
plt.show()


testdata = train_history.history['val_accuracy']
df = pd.DataFrame(testdata)
df.to_csv(r'E:\A_Shi_Chuang\cnn_-frft_-data-master\daochu.csv')


#这里是我导入的测试集的标签表格，用来对比神经网络的测试结果，并且后面生成混淆矩阵；
#这里的标签就是200个测试集的数据的故障标签
file = r"E:\A_Shi_Chuang\Vgg_classfy\label\annotation_fall_170310_1138_02.csv"
all_df = pd.read_csv(file)
ndarray = all_df.values
ndarray[:2]

test_iter = ts_gen()
pres = model.predict_generator(generator=test_iter, steps=math.ceil(520 / Batch_size), verbose=1)
print(pres.shape)

print(ndarray.shape)

ohpres = np.argmax(pres, axis=1)
print(ohpres.shape)
ohpres=ohpres[:200]
ohpres


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

def cm_plot(original_label, predict_label, pic=None):
    cm = confusion_matrix(original_label, predict_label)
    plt.figure()
    plt.matshow(cm, cmap=plt.cm.GnBu)
    plt.colorbar()
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    plt.title('Confusion Matrix')
    if pic is not None:
        plt.savefig(str(pic) + '.jpg')
    plt.show()


plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
cm_plot(ndarray, ohpres)

from sklearn.metrics import accuracy_score
accuracy_score(ndarray, ohpres)

train_history.history['loss']

train_history.history['val_loss']

train_history.history['val_accuracy']

train_history.history['accuracy']
