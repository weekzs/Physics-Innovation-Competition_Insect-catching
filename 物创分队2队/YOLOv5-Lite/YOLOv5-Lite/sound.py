import pygame

file_path = "./有害垃圾.mp3"  # 替换为实际的音频文件路径


pygame.init()
pygame.mixer.init()
pygame.mixer.music.load(file_path)
pygame.mixer.music.play()