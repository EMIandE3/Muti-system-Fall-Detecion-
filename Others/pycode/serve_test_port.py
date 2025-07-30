# -*-coding:utf-8 -*-
# 这个版本能够完成数据的接受，算是初步版本了
import socket
import cv2
import numpy


# 接受图片大小的信息
def recv_size(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def receive_video():
    # socket.AF_INET 用于服务器与服务器之间的网络通信
    # socket.SOCK_STREAM 代表基于TCP的流式socket通信
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 设置地址与端口，如果是接收任意ip对本服务器的连接，地址栏可空，但端口必须设置
    address = ('', 8000)
    s.bind(address)  # 将Socket（套接字）绑定到地址,绑定后，程序就可获得这个端口接收的数据。绑定时，ip为空字符串就默认为自己
    # 使用bind函数时，只能绑定自己电脑的ip和端口！！，空字符串默认为本机ip
    s.listen(True)  # 开始监听TCP传入连接,监听别人的链接，程序就会等待在这里
    print('Waiting for images...')

    # 接受TCP链接并返回（conn, addr），其中conn是新的套接字对象，可以用来接收和发送数据，addr是链接客户端的地址。

    conn, addr = s.accept()  # 接受客户端的身份信息，返回值是一个套接字和客户端的地址端口
    while True:
        # print('Waiting for images...')
        length = recv_size(conn, 16)  # 首先接收来自客户端发送的大小信息
        if isinstance(length, bytes):  # 若成功接收到大小信息，进一步再接收整张图片
            length = length.decode()
            stringData = recv_size(conn, int(length))

            # data = numpy.fromstring(stringData, dtype='uint8')
            data = numpy.frombuffer(stringData, dtype='uint8')  # 这里也对应做了更改，上面一行是原来的，改了之后就没有警告了，在客户端也改了
            decimg = cv2.imdecode(data, cv2.IMREAD_COLOR)

            r_img = decimg.reshape(480, 640, 3)
            cv2.imshow('SERVER', r_img)   # decimg就是图片了

            print('Image received successfully!')


receive_video()  # 直接用它进行调用
