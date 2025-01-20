
import cv2

def play_video(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Failed to open video.")
        return

    # 获取视频的宽度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 获取屏幕尺寸
    screen_width = 2020  # 替换为你的屏幕宽度
    screen_height = 1480  # 替换为你的屏幕高度

    # 计算调整比例
    scale_width = screen_width / width
    scale_height = screen_height / height
    scale = min(scale_width, scale_height)

    # 计算调整后的尺寸
    new_width = int(width * scale/2)
    new_height = int(height * scale/2)

    # 创建窗口并调整尺寸
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', new_width, new_height)

    while cap.isOpened():
        # 读取一帧视频
        ret, frame = cap.read()

        # 检查是否成功读取视频帧
        if not ret:
            print("Error: Failed to read frame.")
            break

        # 在窗口中显示当前帧
        cv2.imshow('Video', frame)

        # 检查是否按下了ESC键，如果是则退出循环
        if cv2.waitKey(25) & 0xFF == 27:
            break

    # 释放视频对象和关闭窗口
    cap.release()
    cv2.destroyAllWindows()

# 播放本地视频文件
if __name__ == '__main__':
    while True:
        video_path = "//home//pi//rubbish//播放视频//WeChat_20240527142742.mp4"
        play_video(video_path)