import torch
from torch.utils.data import Dataset
from python_speech_features import logfbank
import soundfile
import numpy as np
import librosa
import random
import os
import pandas as pd


class SpecDataset(Dataset):
    def __init__(self, wav_scp, label_path, power):
        self.utt2wav_path = {
            x.split()[0].split("/")[-1]: x.split()[0] for x in open(wav_scp)
        }
        self.utt_tuples = self._read_tuples(label_path)
        self.len = len(self.utt_tuples)
        self.power = power

    def _read_tuples(self, label_path):
        utt_tuples = pd.read_csv(label_path)
        return utt_tuples

    def read_wav(self, wav_path):
        try:
            y, sr = soundfile.read(wav_path)
        except:
            raise AssertionError("Unsupported file: %s" % wav_path)
        return y, sr

    def write_wav(self, y, wav_path):
        soundfile.write(wav_path, y, 16000)
        return True

    def _mix_wav(self, y1, y2):
        if len(y1) < len(y2):
            y2 = y2[0 : len(y1)]
            mixed = y1 + y2
        else:
            mixed = np.zeros(len(y1), dtype=y1.dtype)
            mixed[0 : len(y2)] = y2
            mixed += y1
        return mixed

    def _trun_wav(self, y, tlen, offset=0):
        if y.shape[0] < tlen:
            npad = tlen - y.shape[0]
            y = np.pad(y, (0, npad), mode="constant", constant_values=0)
        else:
            y = y[offset : offset + tlen]
        return y

    def _extract_fbank(self, y, sr):
        feat = logfbank(
            y,
            sr,
            # winfunc=np.hamming,
            winlen=0.025,
            winstep=0.01,
            nfilt=64,
            nfft=512,
            lowfreq=0,
            highfreq=None,
            preemph=0.97,
        )
        feat = feat - feat.mean(axis=0)
        return feat.astype("float32")

    def wav2spec(self, y, sr):
        D = librosa.stft(y, n_fft=512, win_length=400, hop_length=160)
        D = np.power(D, self.power)
        mag, phase = np.abs(D), np.angle(D)
        return mag.T, phase.T

    def spec2wav(self, mag, phase):
        mag, phase = mag[: len(phase)], phase[: len(mag)]
        mag, phase = mag.T, phase.T
        D = mag * np.exp(1j * phase)
        D = np.power(D, 1 / self.power)
        y = librosa.istft(D, win_length=400, hop_length=160)
        return y

    def __getitem__(self, sample_idx):
        if isinstance(sample_idx, int):
            index, tlen = sample_idx, None
        elif len(sample_idx) == 2:
            index, tlen = sample_idx
        else:
            raise AssertionError

        wav_utt, algorithm_num = self.utt_tuples.iloc[index]
        wav_path = self.utt2wav_path[wav_utt]

        wav, sr = self.read_wav(wav_path)
        assert sr == 16000

        norm = np.max(np.abs(wav)) * 1.1
        wav = wav / norm

        if tlen != None:
            offset = random.randint(0, max(len(wav) - tlen, 0))
            wav = self._trun_wav(wav, tlen, offset)

        fbank = self._extract_fbank(wav, sr)
        wav_spec, wav_phase = self.wav2spec(wav, sr)

        wav = torch.from_numpy(wav)
        fbank = torch.from_numpy(fbank)
        wav_spec = torch.from_numpy(wav_spec)
        return wav_utt, fbank, wav_spec, wav_phase, wav, algorithm_num

    def __len__(self):
        return self.len
