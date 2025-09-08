'''
 Get video from camera
'''
import cv2
from queue import Queue
import threading  # 导入线程包
import time


# ###########################################通过测试，宽为640，高为480(这两可调节) 帧速率30.0################
def video_cap(my_queue):
    cap = cv2.VideoCapture(1)  # 视频进行读取操作以及调用摄像头
    width = 640
    ret = cap.set(3, width)
    height = 480
    ret = cap.set(4, height)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter('out.avi', fourcc, 20.0, (width, height))

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
            my_queue.put(frame)

            # cv2.imshow(f'{h},{fps}', frame)

        else:
            break

        key = cv2.waitKey(1)
        if key == ord('q'):
            break


def pic_show(my_queue):
    while True:
        pic = my_queue.get()  # 会进入阻塞状态，直到队列中有东西可取为止，真好，正要用这种
        cv2.imshow('hello', pic)
        cv2.waitKey(1)  # 这个函数是刷新屏幕的，因此要显示视频的时候要有他，现在数据是能够正常缓冲的
        # time.sleep(1 / 30.0)  # 延时2s，模拟网络请求


if __name__ == "__main__":
    pic_queue = Queue(maxsize=0)
    # 先创造两个线程
    thread_1 = threading.Thread(target=video_cap, args=(pic_queue,))
    thread_1.start()
    thread_2 = threading.Thread(target=pic_show, args=(pic_queue,))
    thread_2.start()
    print('两个线程正在运行')
