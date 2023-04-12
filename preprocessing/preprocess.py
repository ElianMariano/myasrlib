import numpy as np
import noisereduce as nr

def normalize(audio_signal) -> np.ndarray:
    ratio = 32767 / np.max(np.abs(audio_signal))
    normalized_signal = audio_signal

    for i in range(0, len(normalized_signal)):
        if (normalized_signal.ndim == 1):
            normalized_signal[i] = round(normalized_signal[i]*ratio)
        elif (normalized_signal.ndim == 2):
            normalized_signal[i, 0] = round(normalized_signal[i, 0]*ratio)
            normalized_signal[i, 1] = round(normalized_signal[i, 1]*ratio)
        
    return normalized_signal

def denoise(rate, audio_signal) -> np.ndarray:
    # Coverts the data to mono
    if audio_signal.ndim == 2:
        audio_signal = audio_signal[:, 0]

    # Noise reduced audio
    return nr.reduce_noise(y=audio_signal, sr=rate)