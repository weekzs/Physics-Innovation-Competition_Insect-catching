from test_video import runing,text
import serial
import cv2
import pygame
import numpy as np
import onnxruntime as ort
import time
import random as random
import os

from 播放视频 import play_video
from threading import Thread
import time
from time import sleep


# 自定义的函数，可以替换成其他任何函数
def task(threadName,it):
    # print(f"【线程开始】{threadName}")
    runing(it)
    # print(f"【线程结束】{threadName}")


def task1(threadName,it):
    # print(f"【线程开始】{threadName}")
    non1 = ['anmuxi', 'coca']
    non2 = ['battery', 'medicine']
    non3 = ['apple', 'jiang']
    non4 = ['ganzao', 'paper']
    # 声音加视频
    if it in non1:
        play_video(video_path)
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
    if it in non2:
        play_video(video_path)
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
    if it in non3:
        play_video(video_path)
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
    if it in non4:
        play_video(video_path)
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

    # print(f"【线程结束】{threadName}")


if __name__ == '__main__':
    # result, itt = text()
    thread1 = Thread(target=task, args=("thread_1", itt))  # 线程1：执行任务打印4个a
    thread2 = Thread(target=task1, args=("thread_2", itt))  # 线程2：执行任务打印2个b
    while True:
        result, itt = text()
        thread1.start()  # 线程1开始
        thread2.start()  # 线程2开始

    thread1.join()  # 等待线程1结束
    thread2.join()  # 等待线程2结束
