import matplotlib.pyplot as plt
import cv2

'''
    author: plf
'''


# 根据keras最后的模型对象来画loss图
def plotting(history_object):
    """Function for plotting from training/validation loss from history object
    """
    print(history_object.history.keys())
    plt.plot(history_object.history['loss'], 'b-')
    plt.plot(history_object.history['val_loss'], 'r-')
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


# 输出相关图片到窗口，方便查看模型的输出，输出
def show_img(window_name, img):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 300, 300)
    cv2.imshow(window_name, img)
    cv2.waitKey(3)