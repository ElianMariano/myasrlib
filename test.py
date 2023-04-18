import os
import numpy as np
import warnings
import tensorflow as tf
from tensorflow.python.keras.callbacks import CSVLogger
from audio.read_audio import read_audio_dataset, read_dataset_with_frames
from dataset.dataset_reader import read_file
from models.model import create_model

if __name__ == '__main__':
    # Ignores the warnings thrown by pandas library
    warnings.simplefilter(action='ignore', category=FutureWarning)

    model = tf.keras.models.load_model('save_model/model')

    files = read_file('test_data_sm.csv', root_dir='timit')

    (x, y) = read_dataset_with_frames(files, root_dir=os.path.join('timit', 'data'))

    # val_x = x[0][tf.newaxis, ...]
    # val_y = y[0][tf.newaxis, ...]

    model.evaluate(x, y, batch_size=4)

    # print('Saving Model')
    # model.save('save_model/model')