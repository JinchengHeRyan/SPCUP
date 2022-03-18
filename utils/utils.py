import numpy as np
import sys
import os
import librosa


def get_instance(module, config, *args, **kwargs):
    return getattr(module, config["type"])(*args, **kwargs, **config["args"])


def wav2spec(y, sr, power):
    D = librosa.stft(y, n_fft=512, win_length=400, hop_length=160)
    D = np.power(D, power)
    mag, phase = np.abs(D), np.angle(D)
    return mag.T, phase.T


def spec2wav(mag, phase, power):
    mag, phase = mag.T, phase.T
    D = mag * np.exp(1j * phase)
    D = np.power(D, 1 / power)
    y = librosa.istft(D, win_length=400, hop_length=160)
    return y
