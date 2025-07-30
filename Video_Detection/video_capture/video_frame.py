import cv2
import os


def save_img():
    # 还要改视频的路径
    # video_path = r'D:\My_data\my_pycode\test5_video_capture\video_csi'
    video_path = r'D:\My_data\my_pycode\test5_video_capture\video'
    videos = os.listdir(video_path)  # 找出所有的文件
    print('需要处理的文件为')
    print(videos)
    for video_name in videos:
        file_name = video_name.split('.')[0]  # 只要文件名
        # 一个用于csi采集，一个是视频流采集
        # folder_name = r'D:\My_data\my_pycode\test5_video_capture\frame_csi' + '\\' + file_name
        folder_name = r'D:\My_data\my_pycode\test5_video_capture\frame' + '\\' + file_name
        os.makedirs(folder_name, exist_ok=True)  # 这是新建一个文件夹
        vc = cv2.VideoCapture(video_path + '\\' + video_name)  # 读入视频文件
        c = 0
        rval = vc.isOpened()
        print(rval)
        while rval:  # 循环读取视频帧
            c = c + 1
            ret, frame = vc.read()
            pic_path = folder_name + '/'
            if ret:
                cv2.imwrite(pic_path + file_name + '_' + str(c) + '.jpg', frame)  # 存储为图像,保存名为 文件夹名_数字（第几个文件）.jpg
                cv2.waitKey(1)
            else:
                break
        vc.release()
        print('save_success')
        print(folder_name)


save_img()
