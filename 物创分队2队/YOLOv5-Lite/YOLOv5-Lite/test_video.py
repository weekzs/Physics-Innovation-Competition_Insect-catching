import cv2
import numpy as np
import onnxruntime as ort
import time
import random as random
import os
#from ocr import ocr_train
import serial



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

# def port_close():
#     ser.close()
#     if (ser.isOpen()):
#         print("串口关闭失败！")
#     else:
#         print("串口关闭成功！")
#
#
def send(ser,send_data):
    if (ser.isOpen()):
        ser.write(send_data.encode('utf-8'))  # 编码
        print("发送成功", send_data)
    else:
        print("发送失败！")


os.environ["CUDA_VISIBLE_DEVICES"]="0"




def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
        描述： 在图像 img 上绘制一个边界框，
                 这个函数来自YoLov5项目。
    参数：
        x：一个盒子喜欢 [x1，y1，x2，y2]
        img：OpenCV 映像对象
        color：绘制矩形的颜色，如（0,255,0）
        标签： str
        line_thickness：int
    返回：
        不归路
    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

def _make_grid( nx, ny):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

def cal_outputs(outs,nl,na,model_w,model_h,anchor_grid,stride):
    
    row_ind = 0
    grid = [np.zeros(1)] * nl
    for i in range(nl):
        h, w = int(model_w/ stride[i]), int(model_h / stride[i])
        length = int(na * h * w)
        if grid[i].shape[2:4] != (h, w):
            grid[i] = _make_grid(w, h)

        outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + np.tile(
            grid[i], (na, 1))) * int(stride[i])
        outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(
            anchor_grid[i], h * w, axis=0)
        row_ind += length
    return outs



def post_process_opencv(outputs,model_h,model_w,img_h,img_w,thred_nms,thred_cond):
    conf = outputs[:,4].tolist()
    c_x = outputs[:,0]/model_w*img_w
    c_y = outputs[:,1]/model_h*img_h
    w  = outputs[:,2]/model_w*img_w
    h  = outputs[:,3]/model_h*img_h
    p_cls = outputs[:,5:]
    if len(p_cls.shape)==1:
        p_cls = np.expand_dims(p_cls,1)
    cls_id = np.argmax(p_cls,axis=1)

    p_x1 = np.expand_dims(c_x-w/2,-1)
    p_y1 = np.expand_dims(c_y-h/2,-1)
    p_x2 = np.expand_dims(c_x+w/2,-1)
    p_y2 = np.expand_dims(c_y+h/2,-1)
    areas = np.concatenate((p_x1,p_y1,p_x2,p_y2),axis=-1)
    
    areas = areas.tolist()
    ids = cv2.dnn.NMSBoxes(areas,conf,thred_cond,thred_nms)
    if len(ids)>0:
        return  np.array(areas)[ids],np.array(conf)[ids],cls_id[ids]
    else:
        return [],[],[]

def infer_img(img0,net,model_h,model_w,nl,na,stride,anchor_grid,thred_nms=0.4,thred_cond=0.5):
    # 图像预处理
    img = cv2.resize(img0, [model_w,model_h], interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

    # 模型推理
    outs = net.run(None, {net.get_inputs()[0].name: blob})[0].squeeze(axis=0)

    # 输出坐标矫正
    outs = cal_outputs(outs,nl,na,model_w,model_h,anchor_grid,stride)

    # 检测框计算
    img_h,img_w,_ = np.shape(img0)
    boxes,confs,ids = post_process_opencv(outs,model_h,model_w,img_h,img_w,thred_nms,thred_cond)

    return  boxes,confs,ids

#检测list1中是否有list2的元素
def check_elements_in_list(list1, list2):
    if len(list1) == 0:
        return False, 'q'
    for item in list1:
        if item in list2:
            return True,item
    return False, 'q'




# def text(desired_class):
def text():
    non = ['anmuxi', 'apple', 'battery', 'coca', 'ganzao','jiang','medicine','paper']
    # 模型加载
    model_pb_path = "D://桌面//物创分队2队//YOLOv5-Lite//YOLOv5-Lite//YOLOv5-Lite//runs//train//exp4//weights//best.onnx"  # 修改onnx文件路径
    so = ort.SessionOptions()
    net = ort.InferenceSession(model_pb_path, so)

    # 标签字典
    dic_labels = {0: 'anmuxi', 1: 'apple',
                  2: 'battery', 3: 'coca',
                  4: 'ganzao', 5: 'jiang',
                  6:'medicine',7:'paper',8:'q'}  # 修改为自己的类别

    # desired_labels = [dic_labels[i] for i in desired_classes]

    # 模型参数
    model_h = 320  # 图片resize的大小
    model_w = 320
    nl = 3  # 三层输出对应类别
    na = 3  # 每层3种anchor,对应下方的anchors列表
    stride = [8., 16., 32.]  # 缩放尺度因子
    # 默认anchors大小设置，可以根据自己情况调整
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119],
               [116, 90, 156, 198, 373, 326]]  # 默认anchors大小设置，可以根据自己情况调整
    anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(nl, -1, 2)

    cap = cv2.VideoCapture(0)

    while True:



        success, img0 = cap.read()
        if success:
            # if flag_det:
            t1 = time.time()
            det_boxes, scores, ids = infer_img(img0, net, model_h, model_w, nl, na, stride, anchor_grid,
                                                thred_nms=0.4, thred_cond=0.5)
            jishu = [[] for i in range(len(ids))]
            # print(jishu)
            for j in range(10):
                det_boxes, scores, ids = infer_img(img0, net, model_h, model_w, nl, na, stride, anchor_grid,
                                                thred_nms=0.4, thred_cond=0.5)


                if not det_boxes is None:

                    # print(jishu)
                    for i in range(len(ids)):
                        # jishu[i].append(i)
                        if scores[i]>0.6:
                        # if scores[i] > 0.75:
                            jishu[i].append(ids[i])

                # time.sleep(0.005)

            jishu_list = [[] for i in range(len(ids))]
            # print(jishu[1])
            for i in range(len(ids)):
                # print(jishu_list[i])
                # jishu_list[i] = [arr for arr in jishu[i]]
                # print(jishu_list)
                m = 0
                if len(jishu[i]) >= 7:
                    maxlabel = max(jishu[i], key=jishu[i].count)
                    count = jishu[i].count(maxlabel)
                    m = count
                    jishu_list[i] = maxlabel
                else:
                    jishu_list[i] = 8
                if m < 7:
                    jishu_list[i] = 8
            # print(jishu_list)
            ids = jishu_list
            # print(ids)
            t2 = time.time()
            label = [dic_labels[i] for i in ids]
            successd, item = check_elements_in_list(label, non)
            # print(success)
            # print(item)


            for box, score, id in zip(det_boxes, scores, ids):
                label = '%s:%.2f' % (dic_labels[id], score)

                plot_one_box(box.astype(np.int16), img0, color=(255, 0, 0), label=label, line_thickness=None)

                str_FPS = "FPS: %.2f" % (1. / (t2 - t1))

                cv2.putText(img0, str_FPS, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

            cv2.imshow('usb camera', img0)

            if successd:
                cap.release()
                cv2.destroyAllWindows()
                # print(item)
                return 1, item
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 按q退出
            # cv2.destroyAllWindows("Window")
            break




# if __name__ == '__main__':
#     text();

def runing(it):
    ser = serial.Serial("/dev/ttyAMA0", 115200)  # 打开/dev/ttyAMA10，将波特率配置为115200，其余参数使用默认值
    if ser.isOpen():  # 判断串口是否成功打开
        print("打开串口成功。")
        print(ser.name)  # 输出串口号
        write_len = ser.write("ABCDEFG".encode('utf-8'))

        # # 读取串口输入信息并输出。
        # while True:
        #     com_input = ser.read(1)
        #     if com_input:  # 如果读取结果非空，则输出
        #         print(com_input)
    else:
        print("打开串口失败。")
    non = ['anmuxi', 'apple', 'battery', 'coca', 'ganzao','jiang','medicine','paper']
    non1 = ['anmuxi','coca']
    non2 = ['battery', 'medicine']
    non3 = ['apple','jiang']
    non4 = ['ganzao','paper']

    # s, it =text2(non,nom)
    # print(it)

    # while True :
        # text()
        #----读取串口数据-----------------------------------
    try:
            # while True:
                # result, it = text()
        if it in non1:
            send(ser,'1')
        if it in non2:
            send(ser,'2')
        if it in non3:
            send(ser,'3')
        if it in non4:
            send(ser,'4')
        # print(it)

            # Read_data()						# 前面两行可以注释，换成后面这个函数
    except KeyboardInterrupt:
        if serial != None:
            print("close serial port")
            serial.close()

#
#
