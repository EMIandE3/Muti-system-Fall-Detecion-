# 注意，必须在打开那个显示屏幕的情况下，按下退出按键，才能得到可以打开的视频
import cv2
import datetime
from time import *

cap = cv2.VideoCapture(1)
# print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 构建视频保存的对象
fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'v')  # 为保存视频做准备，构建了一个对象，其中10为帧率，自己可按照需要修改
name_time = str(datetime.datetime.now().hour) + '_' + str(datetime.datetime.now().minute) + '_' + str(datetime.datetime.now().second)
# out = cv2.VideoWriter(fr"D:\My_data\my_pycode\test5_video_capture\video_csi\{name_time}.mp4", fourcc, 30.0, (640,480))
out = cv2.VideoWriter(fr"E:\video_data2\{name_time}.mp4", fourcc, 30.0, (640, 480))

# print(cap.get(3))
# print(cap.get(4))
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:

        # 保存视频的数据集时，注释掉这几行
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # datet = str(datetime.datetime.now())
        # frame = cv2.putText(frame, datet, (20, 150), font, 1.2,  # 第五个参数是控制显示字符的大小，那个元祖是色彩
        #                     (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)  # 显示视频
        out.write(frame)  # 保存视频

        if cv2.waitKey(1) & 0xFF == ord('1'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
