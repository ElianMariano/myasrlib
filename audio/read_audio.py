import librosa
import numpy as np
from preprocessing.preprocess import normalize, denoise

def read_single_audio(file_name, nmels=128, fmax=8000, hop_length=500, options=['noisereduce', 'normalize'], training=False) -> np.ndarray:
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

    return librosa.feature.melspectrogram(y=y, sr=sr, n_mels=nmels, fmax=fmax, hop_length=hop_length)

def read_audio_dataset(audio_files) -> np.ndarray:
    dataset = []

    for single_audio in audio_files:
        dataset.append(read_single_audio(single_audio))

    return dataset