import os
import numpy as np
import warnings
import tensorflow as tf
from audio.read_audio import read_audio_dataset, read_dataset_with_frames
from dataset.dataset_reader import read_file
from models.model import create_model

if __name__ == '__main__':
    # Ignores the warnings thrown by pandas library
    warnings.simplefilter(action='ignore', category=FutureWarning)

    files = read_file('train_data_sm.csv', root_dir='timit')

    (x, y) = read_dataset_with_frames(files, root_dir=os.path.join('timit', 'data'))

    model = create_model(x.shape[1:], y.shape[1])

    opt = tf.keras.optimizers.Adam(learning_rate=0.0625)

    model.compile(
        loss='binary_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    val_x = x[0][tf.newaxis, ...]
    val_y = y[0][tf.newaxis, ...]

    model.fit(x[1:], y[1:], epochs=30, batch_size=25, validation_data=(val_x, val_y))