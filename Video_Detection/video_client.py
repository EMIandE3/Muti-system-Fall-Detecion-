import socket
import cv2
import numpy as np
import time

def send_frame(sock, frame):
    """发送一帧图像：先发送大小，再发送数据"""
    # 对图像进行编码（JPEG格式，质量80）
    result, img_encode = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    
    # 将编码后的图像转为字节流
    data = np.array(img_encode).tobytes()
    
    # 先发送数据长度（固定16字节，不足补空格）
    length = str(len(data)).ljust(16)
    sock.sendall(length.encode('utf-8'))
    
    # 再发送实际图像数据
    sock.sendall(data)

def video_client(server_ip, server_port):
    # 创建TCP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        # 连接服务器
        client_socket.connect((server_ip, server_port))
        print(f"已成功连接到服务器 {server_ip}:{server_port}")
        
        # 打开摄像头（0表示默认摄像头）
        cap = cv2.VideoCapture(0)
        
        # 设置摄像头分辨率（与服务器期望的480x640保持一致）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 检查摄像头是否打开成功
        if not cap.isOpened():
            print("无法打开摄像头")
            return
        
        print("开始传输视频流...（按ESC键停止）")
        fps_counter = 0
        start_time = time.time()
        
        while True:
            # 读取一帧图像
            ret, frame = cap.read()
            if not ret:
                print("无法获取图像帧")
                break
            
            # 发送帧到服务器
            send_frame(client_socket, frame)
            
            # 计算并显示帧率
            fps_counter += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1:
                fps = fps_counter / elapsed_time
                print(f"帧率: {fps:.1f} FPS", end='\r')
                fps_counter = 0
                start_time = time.time()
            
            # 显示本地采集的图像（可选）
            cv2.imshow('本地摄像头', frame)
            
            # 按ESC键退出
            if cv2.waitKey(1) == 27:
                print("\n用户终止传输")
                break
            
            # 适当延迟，控制传输速率（可选）
            # time.sleep(0.01)
            
    except ConnectionRefusedError:
        print(f"无法连接到服务器 {server_ip}:{server_port}，请检查服务器是否已启动")
    except Exception as e:
        print(f"传输过程中发生错误: {str(e)}")
    finally:
        # 释放资源
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        client_socket.close()
        cv2.destroyAllWindows()
        print("客户端已关闭")

if __name__ == '__main__':
    # 服务器IP和端口（根据实际情况修改）
    SERVER_IP = '127.0.0.1'  # 本地测试用
    # SERVER_IP = '192.168.1.100'  # 实际部署时改为服务器的IP地址
    SERVER_PORT = 8000
    
    video_client(SERVER_IP, SERVER_PORT)
