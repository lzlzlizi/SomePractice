import cv2
import numpy as np


# 可以将PIL图像转化为CV2图像
def pil2cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2YUV)  # 这里可根据需求修改
