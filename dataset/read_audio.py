import librosa
import numpy as np
import tensorflow as tf
import os
import re
import pandas as pd
from preprocess import normalize, denoise, padding, resize, specmag, specnorm, random_pitch, white_noise
from dataset.label_reader import read_label_classes, read_label_sparse
from tqdm import tqdm
from random import randint
from config import read_config

# TODO in the future use this filter
def predicate(timestamps, most_shown_labels=['h#', 's', 'n', 'l', 'r', 'iy', 'ih', 'ix', 'dcl', 'kcl', 'tcl']) -> np.array:
    PROB = True if randint(0, 2) == 1 else False

    newTimestamps = np.empty((0, timestamps.shape[1]))
    for i in range(0, len(timestamps)):
        if timestamps[i][2] in most_shown_labels:
            if PROB:
                newTimestamps = np.concatenate((newTimestamps, np.reshape(timestamps[i], (1, timestamps.shape[1]))), axis=0)
        else:
            newTimestamps = np.concatenate((newTimestamps, np.reshape(timestamps[i], (1, timestamps.shape[1]))), axis=0)
    return newTimestamps

def read_single_audio(file):
    config = read_config()
    WIDTH = int(config['width'])
    HEIGHT = int(config['height'])
    FRAME_LENGTH = int(config['frame_length'])
    FRAME_STEP = int(config['frame_step'])
    PREPROCESSING = config['preprocessing'].split(',')
    AUGMENTATION = True if config['augmentation'] == 'true' else False

    phn_file = re.sub(r"(\.wav)", '.PHN', file)

    timestamps = pd.read_csv(phn_file, ' ', header=None).to_numpy()

    classes = read_label_classes()

    Y = tf.constant([])

    for i in range(0, len(timestamps)):
        Y = tf.concat([Y, [classes.index(timestamps[i][2])]], axis=0)

    X = []

    y, sr = librosa.load(file)

    # data augmentation
    if AUGMENTATION:
        y = white_noise(y)

        y = random_pitch(y, sr)

    for i in range(0, len(timestamps)):
        frame = y[timestamps[i][0]:timestamps[i][1]]

        # For this dataset
        if len(frame) < 10:
            frame = np.concatenate((frame, np.array([0]*(10-len(frame)), dtype=np.float32)), axis=0)

        spectrogram = tf.signal.stft(frame, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP)

        if 'specmag' in PREPROCESSING:
            spectrogram = specmag(spectrogram)
        
        if 'specnorm' in PREPROCESSING:
            spectrogram = specnorm(spectrogram)
        
        if 'resize' in PREPROCESSING:
            spectrogram = resize(spectrogram, width=WIDTH, height=HEIGHT)

        if (len(X) == 0):
            X = spectrogram
        else:
            X = tf.concat([X, spectrogram], axis=0)

    return (X, Y)

def read_dataset(files, root_dir):
    x_train = []
    y_train = []

    for audio in tqdm(files):
        X, Y = read_single_audio(os.path.join(root_dir, audio))
        
        if (len(x_train) == 0):
            x_train = X
        else:
            x_train = tf.concat([x_train, X], axis=0)

        if (len(y_train) == 0):
            y_train = Y
        else:
            y_train = tf.concat([y_train, Y], axis=0)

    return (x_train, y_train)