import librosa
import numpy as np
import tensorflow as tf
import os
import re
import pandas as pd
from preprocessing.preprocess import normalize, denoise, padding, resize
from dataset.label_reader import read_label_classes, read_label_sparse
from tqdm import tqdm

def read_single_audio(file_name, nmels=128, fmax=8000, hop_length=500, options=['noisereduce', 'normalize', 'padding'], training=False, maxwidth=500000) -> tf.Tensor:
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
        if `False`, skips the preprocessing phase

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
    
    if 'padding' in options and not training:
        y = padding(y, maxwidth=maxwidth)
    
    audio = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=nmels, fmax=fmax, hop_length=hop_length)
    audio = tf.convert_to_tensor(audio, dtype=tf.float32)[tf.newaxis, ...]

    if 'resize' in options and not training:
        audio = resize(audio)

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

    for single_audio in tqdm(audio_files):
        if dataset == None:
            audio = read_single_audio(os.path.join(root_dir, single_audio))
            dataset = audio[tf.newaxis, ...]
        else:
            audio = read_single_audio(os.path.join(root_dir, single_audio))[tf.newaxis, ...]
            dataset = tf.concat([dataset, audio], axis=0)

    return dataset

def read_audio_with_frames(y, sr, timestamps, nmels=128, fmax=8000, hop_length=500, options=['noisereduce', 'normalize', 'resize'], training=False, maxwidth=100000) -> tf.Tensor:
    """
        Works the same way as the function read_single_audio, but the only difference is the audio comes
        divided according with specific label timestamps.
        Parameters
        -----------
        y : np.ndarray [shape=(n,) or (..., n)]
            audio time series. Multi-channel is supported.
        
        sr : number > 0 [scalar]
            sampling rate of ``y``

        timestamps : np.ndarray [shape=(n,) or (..., n)]
            for each element should exist a timestamp and a label.
            Example: [start, end, label]
        
        options : list [noisereduce | normalize]
            List containing the operations to peform in preprocessing
        
        training: boolean
            if `False`, skips the preprocessing phase.
        
        maxwidth: int
            The max width variable is used to apply padding with 80000 default size

        Returns
        -------
        S : np.ndarray [shape=(..., n_mels, t)]
            Mel spectrogram
    """
    audio_frames = tf.constant([], dtype=tf.float32)

    audio_series = y

    if 'noisereduce' in options and not training:
        audio_series = denoise(sr, y)
    
    if 'normalize' in options and not training:
        audio_series = normalize(audio_series)

    for i in timestamps:
        # Selects the audio frame
        audio = audio_series[i[0]:i[1]]

        if 'padding' in options and not training:
            audio = padding(audio, maxwidth=maxwidth)

        # Spectrogram
        audio = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=nmels, fmax=fmax, hop_length=hop_length)
        audio = tf.convert_to_tensor(audio, dtype=tf.float32)[tf.newaxis, ...]

        # print(audio.shape)
        if 'resize' in options and not training:
            audio = resize(audio)

        # Reshape the spectrogram
        # audio = np.reshape(audio, (1, audio.shape[0], audio.shape[1]))

        if len(audio_frames) == 0:
            audio_frames = audio
        else:
            # audio_frames = np.concatenate((audio_frames, audio), axis=0)
            audio_frames = tf.concat([audio_frames, audio], axis=0)

    return audio_frames

def read_dataset_with_frames(audio_files, root_dir='', label_extension='.PHN', sparse=True):
  """ Returns the whole dataset divided by frames and the corresponding label for each frame
  """
  input, output = np.array([]), np.array([])

  for file in tqdm(audio_files):
    label_file = re.sub(r"(\.WAV)(\.wav)", label_extension, file)

    timestamps = pd.read_csv(os.path.join(root_dir, label_file), ' ', header=None).to_numpy()

    classes = read_label_classes()

    y, sr = librosa.load(os.path.join(root_dir, file))

    audio = read_audio_with_frames(y, sr, timestamps) # Test with resizing

    if len(input) == 0:
      input = audio
    else:
      input = np.concatenate((input, audio), axis=0)

    if sparse:
       if len(output) == 0:
           output = read_label_sparse(os.path.join(root_dir, label_file), classes)
       else:
           output = tf.concat([output, read_label_sparse(os.path.join(root_dir, label_file), classes)], axis=0)
    else:
       # Create a prob matrix for the timestamps
        prob_timestamps = np.zeros((timestamps.shape[0], len(classes)), dtype=np.int32)

        for i in range(0, len(timestamps)):
            prob_timestamps[i][classes.index(timestamps[i][2])] = 1

        if len(output) == 0:
            output = prob_timestamps
        else:
            output = np.concatenate((output, prob_timestamps), axis=0)

        output = tf.convert_to_tensor(output, dtype=tf.int32)

  return (input, output)