import cv2
import numpy as np
import onnxruntime as ort
import time

import os
import serial
from test_video import text
from ocr import ocr_train


os.environ["CUDA_VISIBLE_DEVICES"]="0"

import serial.tools.list_ports

# 获取所有串口设备实例。
# 如果没找到串口设备，则输出：“无串口设备。”
# 如果找到串口设备，则依次输出每个设备对应的串口号和描述信息。
def saomiao():
    ports_list = list(serial.tools.list_ports.comports())
    if len(ports_list) <= 0:
        print("无串口设备。")
    else:
        print("可用的串口设备如下：")
        for comport in ports_list:
            print(list(comport)[0], list(comport)[1])
    # isOpen()函数来查看串口的开闭状态

def port_close():
    ser.close()
    if (ser.isOpen()):
        print("串口关闭失败！")
    else:
        print("串口关闭成功！")


def send(send_data):
    if (ser.isOpen()):
        ser.write(send_data.encode('utf-8'))  # 编码
        print("发送成功", send_data)
    else:
        print("发送失败！")



if __name__ == '__main__':
    # 串口
    # port_open_recv()
    saomiao()
    ser = serial.Serial('COM13', 115200, bytesize=8, parity='N', stopbits=1, timeout=1)
    if (ser.isOpen()):
        print("串口打开成功！")
    else:
        print("串口打开失败！")

    send('0')
    while True:
        try:
            count = ser.inWaiting()   #获得input buffer中缓存字节数
            if count > 0:
                # 初始化数据
                Read_buffer = []
                # 接收数据至缓存区
                Read_buffer = ser.readline()  # 我们需要读取的是40个寄存器数据，即40个字节
                send('1')
                print(f'{Read_buffer}')
                if Read_buffer == b'1':
                    print('2')
        except KeyboardInterrupt:
            if serial != None:
                print("close serial port")
                serial.close()
    # # non1 = ['red_box', 'wangzai_milk_can', 'coca_cola_can', 'dongpeng', 'lays_crisps', 'mengniu_milk']
    # # nom1 = [1, 1, 1, 1, 1, 1]
    # non = ['red_box', 'wangzai_milk_can', 'coca_cola_can', 'dongpeng', 'lays_crisps', 'mengniu_milk']
    # nom = [1, 1, 1, 1, 1, 1]
    #
    # while True :
    #     if sum(nom)<=0:
    #         send('s')
    #     # ----读取串口数据-----------------------------------
    #     try:
    #         count = ser.inWaiting()   #获得input buffer中缓存字节数
    #         if count > 0:
    #             # 初始化数据
    #             Read_buffer = []
    #             # 接收数据至缓存区
    #             Read_buffer = ser.readline()  # 我们需要读取的是40个寄存器数据，即40个字节
    #             # send('1')
    #             # print(Read_buffer)
    #             if Read_buffer == 'a':
    #                 non, nom = ocr_train()
    #                 time.sleep(0.2)
    #                 send('o')
    #                 Read_buffer = []
    #             if Read_buffer == 'b':
    #                 time.sleep(0.2)
    #                 result, item = text(non, nom)
    #                 if result == 1:
    #                     i = non.index(item)
    #                     nom[i] = nom[i] - 1
    #                 send(result)
    #                 Read_buffer = []
    #
    #             # Read_data()						# 前面两行可以注释，换成后面这个函数
    #     except KeyboardInterrupt:
    #         if serial != None:
    #             print("close serial port")
    #             serial.close()
