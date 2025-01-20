import pyttsx3
engine = pyttsx3.init()  # 创建engine并初始化
engine.say("有害垃圾")
engine.setProperty('rate', 125)
engine.runAndWait()  # 等待语音播报完毕
rate = engine.getProperty('rate') # 获取当前语速的详细信息
print(rate)  # 打印当前语速


engine.say("其他垃圾")
engine.setProperty('rate', 125)
engine.runAndWait()  # 等待语音播报完毕
rate = engine.getProperty('rate') # 获取当前语速的详细信息
print(rate)  # 打印当前语速


engine.say("可回收垃圾")
engine.setProperty('rate', 125)
engine.runAndWait()  # 等待语音播报完毕
rate = engine.getProperty('rate') # 获取当前语速的详细信息
print(rate)  # 打印当前语速
