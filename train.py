import os
import numpy as np
import warnings
import tensorflow as tf
from tensorflow.python.keras.callbacks import CSVLogger
from audio.read_audio import read_audio_dataset, read_dataset_with_frames
from dataset.dataset_reader import read_file
from models.model import create_model
from config import read_config

if __name__ == '__main__':
    # Ignores the warnings thrown by pandas library
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    
    config = read_config()
    SPARSE = True if config['sparse'] == 'true' else False
    EPOCH = int(config['epochs'])
    BATCH_SIZE = int(config['batch_size'])
    TRAINING_FILES = config['training_files'].split(',')

    test_files = read_file('test_data_sm.csv', root_dir='timit')

    print('Loading testing data')
    (x_test, y_test) = read_dataset_with_frames(test_files, root_dir=os.path.join('timit', 'data'), sparse=SPARSE)

    # model = create_model(x_train.shape[1:], y_train.shape[0])
    # model = create_model(x_train.shape[1:], y_train.shape[1])
    model = create_model(x_test.shape[1:], y_test.shape[1])

    opt = tf.keras.optimizers.Adam(learning_rate=625e-6)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    csv_logger = CSVLogger('training.csv', ',', True)

    for file in TRAINING_FILES:
        train_files = read_file(file, root_dir='timit')

        print('Loading training data')
        (x_train, y_train) = read_dataset_with_frames(train_files, root_dir=os.path.join('timit', 'data'), sparse=SPARSE)

        model.fit(x_train, y_train, epochs=EPOCH, batch_size=BATCH_SIZE, callbacks=[csv_logger], validation_data=(x_test, y_test))

    print('Saving Model')
    model.save('save_model/model')