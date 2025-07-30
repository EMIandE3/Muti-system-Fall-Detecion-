# 这段代码的主要目的是从一个视频文件夹中提取每一帧，并将每一帧图片作为输入传递给一个预训练的VGG16模型进行特征提取
import os
import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset

from torch.utils.data import DataLoader,Dataset
from torchvision import transforms as T
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models

class VGGNet_Transfer(nn.Module):
    def __init__(self, num_classes=10):	   #num_classes，此处为 二分类值为2
        super(VGGNet_Transfer, self).__init__()
        net = models.vgg16(pretrained=True)   #从预训练模型加载VGG16网络参数
        net.classifier = nn.Sequential()	#将分类层置空，下面将改变我们的分类层
        self.features = net		#保留VGG16的特征层
        self.classifier = nn.Sequential(    #定义自己的分类层
                nn.Linear(512 * 7 * 7, 512),  #512 * 7 * 7不能改变 ，由VGG16网络决定的，第二个参数为神经元个数可以微调
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 128),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(128, num_classes),
        )
 
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 1. prepare data

#提取视频中图片 按照每帧提取
video_path = r'./video/' #视频所在的路径
f_save_path = './pic/' #保存图片的上级目录
videos = os.listdir(video_path) #返回指定路径下的文件和文件夹列表。
for video_name in videos:  #依次读取视频文件
    file_name = video_name.split('.')[0] #拆分视频文件名称 ，剔除后缀
    folder_name = f_save_path + file_name #保存图片的上级目录+对应每条视频名称 构成新的目录存放每个视频的
    os.makedirs(folder_name,exist_ok=True) #创建存放视频的对应目录
    vc = cv2.VideoCapture(video_path+video_name) #读入视频文件
    c=0   #计数 统计对应帧号
    rval=vc.isOpened() #判断视频是否打开 返回True或Flase
    while rval: #循环读取视频帧
        rval, frame = vc.read() #videoCapture.read() 函数，第一个返回值为是否成功获取视频帧，第二个返回值为返回的视频帧：
        pic_path = folder_name+'/'
        print(pic_path)
        if rval:
            cv2.imwrite(pic_path + file_name + '_' + str(c) + '.jpg', frame) #存储为图像,保存名为 文件夹名_数字（第几个文件）.jpg
            
            model = VGGNet_Transfer(num_classes=10)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            output = model(frame) 
            print(output)

            # print(pic_path + file_name + '_' + str(c) + '.jpg')
            cv2.waitKey(1000) #waitKey()--这个函数是在一个给定的时间内(单位ms)等待用户按键触发;如果用户没有按下 键,则接续等待(循环)
            c = c + 1
        else:
            break
    vc.release()
    print('save_success'+folder_name)

# 2. load model


