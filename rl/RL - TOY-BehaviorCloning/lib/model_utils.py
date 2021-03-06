'''
    1) help you to create training data, validation data and testing data
    2) help you to plot 'mse'
'''

import cv2
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os


class ReadData(object):
    """
    this Class is for processing image.
                    1. read data from your log.
                    2. shuffle data to create "training" and "testing" data.
                    3. crip data, make less noisy
    """

    def __init__(self):
        """
        Initialization
        """

        self.data_len = 0

        self.total_log = pd.DataFrame()
        self.train_log = pd.DataFrame()
        self.valid_log = pd.DataFrame()
        self.test_log = pd.DataFrame()

        self.train_data = 0
        self.valid_data = 0

        self.test_data = 0
        self.batch_size = 64

    def _process_log(self):
        """
        split the total data into 'train data', 'validation data' and 'test data'
                        70% used for training
                        20% used for validation
                        10% used for testing
        """

        left_index = int(self.data_len * 0.70)
        middle_index = int(self.data_len * 0.90)

        self.total_log = shuffle(self.total_log)
        self.train_log = self.total_log.iloc[:left_index, :]
        self.valid_log = self.total_log.iloc[left_index:middle_index, :]
        self.test_log = self.total_log.iloc[middle_index:, :]

        print('train size:{}'.format(len(self.train_log)))
        print('valid size:{}'.format(len(self.valid_log)))
        print('test size:{}'.format(len(self.test_log)))

    def _image_process(self, img, current_angle):
        """Pre process images before fed into model"""
        # crop image
        # self.show_img('origin', img)
        #img = img[60:130, :, :]
        # self.show_img('after crop', img)
        # apply image augmentation techniques

        img, current_angle = self._adjust_images(img, current_angle)
        # apply GaussianBlur
        #if np.random.rand() > 0.5:
        blur = np.random.randint(0, 3)
        if blur == 1:
            img = cv2.GaussianBlur(img, (3, 3), 0)
        # change image from BGR to YUV
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        # self.show_img('finally', img)
        return img, current_angle

    # @staticmethod
    def _adjust_images(self, image, current_angle):
        """apply random brightness/random flip and random shear to images
        :param image: image array from cv2 imread function
        :param current_angle: angle value from driving log
        :return: adjusted image and angle value
        """
        bright = np.random.randint(0, 3)
        if bright == 1:
            cv2.add(image, np.array([np.random.uniform(-100, 100)]))
        flip_flag = np.random.randint(0, 2)
        if flip_flag == 1:
            image = np.fliplr(image)
            current_angle = current_angle * (-1.)

        return image, current_angle

    def _create_image(self, log, is_train):
        """
                        create image and label.

                        Args:
                                        log : 	contain dir for image
                                        is_train: 1 means is training data, 0 otherwise

                        Returns:
                                        data sets and labels.
        """

        sizes = log.shape[0]
        while True:
            shuffle(log)
            for offset in range(0, sizes, self.batch_size):
                '''
                if is_train:
                    print('train offset/sizes:', offset/sizes*100, '%')
                else:
                    print('val offset/sizes:', offset/sizes*100, '%')
                '''

                batch_sample = log.iloc[offset:offset + self.batch_size, :]
                images = []
                angels = []

                for index, row in batch_sample.iterrows():
                    position = np.sum(
                        [np.random.randint(0, 3) if is_train == 1 else 0])

                    image_path = row[position]
                    angel = float(row[3]) + \
                        np.sum([0.0 if position == 0 else 0.0]) + \
                        np.sum([0.2 if position == 1 else 0.0]) + \
                        np.sum([-0.2 if position == 2 else 0.0])

                    image = cv2.imread(image_path)
                    if image is not None:
                        image, angle = self._image_process(image, angel)
                        images.append(image)
                        angels.append(angel)

                    else:
                        pass
                        # print(image_path, 'file is none')

                x_data = np.array(images)
                y_data = np.array(angels)
                yield shuffle(x_data, y_data)  # 将x,y 对应位置随机

    def read_csv_data(self, log_dir):
        
        self.total_log = pd.read_csv(log_dir,header=None)
        self.data_len = self.total_log.shape[0]

        self._process_log()

    def handle_data(self):
        self.train_data = self._create_image(self.train_log, is_train=1)
        self.valid_data = self._create_image(self.valid_log, is_train=0)
        self.test_data = self.prepare_test_data(self.test_log)

    
    # @staticmethod
    def plotting(self, history_object):
        """Function for plotting from training/validation loss from history object
        """

        plt.plot(history_object.history['loss'],'b-')
        plt.plot(history_object.history['val_loss'],'r-')
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.savefig('loss_curve.jpg')

    # @staticmethod
    def prepare_test_data(self, log):
        """prepare test sets data for our model
        :param log: driving log to provide test data
        """
        center_image_names = log[0]
        angle = log[3]
        images = []
        for index, image_path in center_image_names.iteritems():
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)
        images = np.array(images)
        adjust_images = []
        for img in images:
            #img = img[60:130, :, :]
            # resize image
            # change image from BGR to YUV
            #img = cv2.GaussianBlur(img, (3, 3), 0)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
            adjust_images.append(img)
            angle = np.array(angle)
        return np.array(adjust_images), np.array(angle)
