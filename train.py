import os
import numpy as np
import warnings
import tensorflow as tf
from tensorflow.python.keras.callbacks import CSVLogger
from audio.read_audio import read_audio_dataset, read_dataset_with_frames
from dataset.dataset_reader import read_file
from models.model import create_model

# TODO Fix labels for sparse categorical entropy
if __name__ == '__main__':
    # Ignores the warnings thrown by pandas library
    warnings.simplefilter(action='ignore', category=FutureWarning)

    files = read_file('train_data_sm.csv', root_dir='timit')

    (x, y) = read_dataset_with_frames(files, root_dir=os.path.join('timit', 'data'))

    # TODO Only for sparse categorical cross entropy loss function
    y = y.numpy()
    y = np.fromiter(map(lambda a: np.where(a == a.max())[0][0], y), dtype=np.int32)
    y = tf.convert_to_tensor(y, dtype=tf.int32)

    print(y[0:30]) # 0, 11, 42, 26
    print(y[0])

    model = create_model(x.shape[1:], y.shape[0])  # TODO For sparse categorical cross entropy 
    # model = create_model(x.shape[1:], y.shape[1])

    opt = tf.keras.optimizers.Adam(learning_rate=0.0625e-6)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    val_x = x[0][tf.newaxis, ...]
    # val_y = y[0][tf.newaxis, ...]
    val_y = y[0] # TODO For sparse categorical cross entropy

    csv_logger = CSVLogger('training.csv', ',', True)

    model.fit(x[1:], y[1:], epochs=30, batch_size=20, callbacks=[csv_logger]) # , validation_data=(val_x, val_y)

    print('Saving Model')
    model.save('save_model/model')