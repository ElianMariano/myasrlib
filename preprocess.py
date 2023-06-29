import numpy as np
import noisereduce as nr
import tensorflow as tf

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

def padding(y, maxwidth=20000) -> np.ndarray:
    zero_padding = np.zeros(maxwidth - len(y))

    audio = np.concatenate((zero_padding, y), axis=0)

    return audio

def resize(spectrogram, width, height) -> tf.Tensor:
    return tf.image.resize(spectrogram[..., tf.newaxis], [width, height], method='nearest')[tf.newaxis, ...]

def specnorm(spectrogram) -> tf.Tensor:
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)

    return spectrogram

def specmag(spectrogram) -> tf.Tensor:
    """Filters only the magnitude"""
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)

    return spectrogram

def white_noise(audio) -> tf.Tensor:
    noise_factor = random.uniform(0.001, 0.01)
    white_noise = np.random.randn(len(audio)) * noise_factor

    return audio + white_noise

def random_pitch(audio, sr) -> np.ndarray:
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=random.uniform(-10, 10))