import librosa
import numpy as np
import tensorflow as tf
import os
from preprocessing.preprocess import normalize, denoise
# from dataset.label_reader import read_label_classes, read_label

def read_single_audio(file_name, nmels=128, fmax=8000, hop_length=500, options=['noisereduce', 'normalize'], training=False, maxwidth=80000, classes=[]) -> tf.Tensor:
    """Reads a single audio file and peforms basic preprocessing operations if necessary.

    Parameters
    -----------
    filename : string 
        File name or path to the audio file.
    nmels : int
        Length of fft window 
    fmax : int
        Max frequency length
    hop_length : int
        number of samples between successive frames
    options : list [noisereduce | normalize]
        List containing the operations to peform in preprocessing
    training: boolean
        if `False`, skips the preprocessing phase,
    maxwidth: int
        The max width variable is used to apply padding with 80000 default size

    Returns
    -------
    S : np.ndarray [shape=(..., n_mels, t)]
        Preprocessed mel spectrogram
    """
    y, sr = librosa.load(file_name)

    if 'noisereduce' in options and not training:
        y = denoise(sr, y)
    
    if 'normalize' in options and not training:
        y = normalize(y)

    # If there are classes for audio, return a framed audio
    if len(classes) != 0:
      result = []
      for i in classes:
        frame = y[i[0]:i[1]]
        zero_padding = np.zeros(maxwidth - frame.shape[0])

        frame = np.concatenate((frame, zero_padding), axis=0)

        frame = np.reshape(frame, (1, frame.shape[0]))

        if len(result) == 0:
          result = frame
        else:
          result = np.concatenate([result, frame], axis=0)

      return tf.convert_to_tensor(result, dtype=tf.int32)
    else:
      zero_padding = np.zeros(maxwidth - y.shape[0])

      y = np.concatenate((y, zero_padding), axis=0)
      
      audio = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=nmels, fmax=fmax, hop_length=hop_length)

      return tf.convert_to_tensor(audio, dtype=tf.int32)

def read_audio_dataset(audio_files, root_dir='', withlabels=False) -> tf.Tensor:
    """Reads an array containing a path of audio files and returns a tensor for each audio file.

    Parameters
    -----------
    audio_files : string 
        File name or path to the audio file.

    Returns
    -------
    S : tf.Tensor [shape=()]
        Audio files
    """
    dataset = None

    for single_audio in audio_files:
        if dataset == None:
            audio = read_single_audio(os.path.join(root_dir, single_audio))
            dataset = audio[tf.newaxis, ...]
        else:
            audio = read_single_audio(os.path.join(root_dir, single_audio))[tf.newaxis, ...]
            dataset = tf.concat([dataset, audio], axis=0)

    return dataset