import utils.autoanchor as autoAC

# 对数据集重新计算 anchors
if __name__ == '__main__':
    new_anchors = autoAC.kmean_anchors('G:/Visual items/YOLOv5-Lite/YOLOv5-Lite/YOLOv5-Lite/data/mydata.yaml', 21, 640, 5.0, 1000, True)
    print(new_anchors)