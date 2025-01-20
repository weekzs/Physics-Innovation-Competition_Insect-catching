import serial
ser = serial.Serial("/dev/ttyAMA0", 115200)    # 打开/dev/ttyAMA10，将波特率配置为115200，其余参数使用默认值
if ser.isOpen():                        # 判断串口是否成功打开
    print("打开串口成功。")
    print(ser.name)    # 输出串口号

    write_len = ser.write("ABCDEFG".encode('utf-8'))

    # 读取串口输入信息并输出。
    while True:
        com_input = ser.read(1)
        if com_input:  # 如果读取结果非空，则输出
            print(com_input)

else:
    print("打开串口失败。")