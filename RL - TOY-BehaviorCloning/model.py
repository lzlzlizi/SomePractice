import json
import matplotlib.pyplot as plt
import numpy as np

from lib.model_utils import *
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras import initializers
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import time
import os

class Model(ReadData):
    """
                                    your Model
    """

    def __init__(self):
        ReadData.__init__(self)
        self.batch_size = 64
        self.save_model_path = 'CarND-Behavioral-Cloning-P3/model.h5'
        
        
    @property
    def nvidia_model(self):
        """
               model : https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
        """
        
        '''
        your model
        '''
        drop_prob = 0.25
        model = Sequential()
        model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3)))
        model.add(Lambda(lambda x: (x / 255.0) - 0.5))
        model.add(Conv2D(24,5,5, border_mode='valid', activation = "elu", subsample = (2,2),  use_bias=False))
        model.add(BatchNormalization())
        model.add(Conv2D(36,5,5, border_mode='valid', activation = "elu", subsample = (2,2),use_bias=False))
        model.add(BatchNormalization())
        model.add(Conv2D(48,5,5, border_mode='valid', activation = "elu", subsample = (2,2),use_bias=False))
        model.add(BatchNormalization())
        model.add(Conv2D(64,3,3, border_mode='valid', activation = "elu", subsample = (1,1),use_bias=False))
        model.add(BatchNormalization())
        model.add(Conv2D(64,3,3, border_mode='valid', activation = "elu", subsample = (1,1),use_bias=False))
        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Dense(128, activation = 'elu', use_bias = False))
        model.add(Dropout(drop_prob))
        model.add(BatchNormalization())
        model.add(Dense(64, activation = 'elu', use_bias = False))
        model.add(Dense(10, activation = 'elu'))
        model.add(Dense(1))
        model.summary()
    
        return model

    def build_and_train_model(self):
        self.model = self.nvidia_model
        self.model.compile(loss='mse', optimizer=Adam() )
        checkpointer = ModelCheckpoint(filepath="v2-weights.{epoch:02d}-{val_loss:.2f}.h5", verbose=1, save_best_only=False)
        early = EarlyStopping(patience=5)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, mode='auto',min_lr = 10e-6)

        his = self.model.fit_generator(self.train_data, steps_per_epoch=int(len(self.train_log)/self.batch_size), 
                                     validation_data=self.valid_data,use_multiprocessing=True, validation_steps = int(len(self.valid_log)/self.batch_size), 
                                     nb_epoch=32, callbacks=[reduce_lr, early, checkpointer])
        self.model.save(self.save_model_path)
        return his
        #########################################################
        # build your model and train






def main():
    """Main function of the model
    """
    ################################
    # change thi
    log_dir = "/home/lzlzlizi/Documents/data/autodrive/driving_log.csv"
    

    my_model = Model()
    my_model.read_csv_data(log_dir)
    my_model.handle_data()
    begin_time = time.time()
    his = my_model.build_and_train_model()
    end_time = time.time()
    print('训练时间:', (end_time - begin_time) / 60, 'min')
    # eval
    print(my_model.model.evaluate(my_model.test_data[0],my_model.test_data[1],batch_size=256))

    my_model.plotting(his)


if __name__ == "__main__":
    main()
