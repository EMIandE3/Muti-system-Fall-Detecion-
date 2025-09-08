'''
 Get video from camera
'''
import cv2
from queue import Queue
import threading  # 导入线程包


# ###########################################通过测试，宽为640，高为480(这两可调节) 帧速率30.0################
def video_cap():
    cap = cv2.VideoCapture(1)  # 视频进行读取操作以及调用摄像头
    width = 640
    ret = cap.set(3, width)
    height = 480
    ret = cap.set(4, height)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter('out.avi', fourcc, 30.0, (width, height))

    while cap.isOpened():  # 判断视频读取或者摄像头调用是否成功，成功则返回true。
        ret, frame = cap.read()
        if ret is True:
            print('frame shape:', frame.shape)
            frame = cv2.resize(frame, (640, 480))
            # w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获得视频流的宽度
            # h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获得视频流的高度
            h = frame.size
            fps = cap.get(cv2.CAP_PROP_FPS)  # 获得帧速率
            out.write(frame)

            cv2.imshow(f'{h},{fps}', frame)

        else:
            break

        key = cv2.waitKey(1)
        if key == ord('q'):
            break


if __name__ == "__main__":
    video_cap()