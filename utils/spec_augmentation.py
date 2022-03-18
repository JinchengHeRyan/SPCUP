import soundfile
import os
import random
import numpy as np


def read_wav(self, wav_path):
    try:
        y, sr = soundfile.read(wav_path)
    except:
        raise AssertionError("Unsupported file: %s" % wav_path)
    return y, sr


def write_wav():
    pass


def spec_mask(self, feat, freq_flag=True, time_flag=True):
    if freq_flag:
        f_wid = randint(0, 8)
        f_beg = randint(0, 84 - f_wid)
        feat[:, f_beg : f_beg + f_wid] = 0
    if time_flag:
        t_wid = randint(0, 4)
        t_beg = randint(0, 32 - t_wid)
        feat[t_beg : t_beg + t_wid, :] = 0
    return feat


def spec_norm(self, feat):
    fea_mean = feat.mean()
    fea_std = feat.std()
    f_wid = randint(0, 8)
    f_beg = randint(0, 84 - f_wid)
    feat[:, f_beg : f_beg + f_wid] = np.random.normal(fea_mean, fea_std, (32, f_wid))
    t_wid = randint(0, 12)
    t_beg = randint(0, 32 - t_wid)
    feat[t_beg : t_beg + t_wid, :] = np.random.normal(fea_mean, fea_std, (t_wid, 84))
    return feat


def spec_swap(self, feat, label):
    f_wid = randint(0, 8)
    f_beg = randint(0, 84 - f_wid)

    extra_id = randint(0, len(self.label2utts[label]) - 1)
    extra_feat_path = self.utt2data_dict[self.label2utts[label][extra_id]]

    signal, sr = sf.read(extra_feat_path)
    extra_feat = self._transform_data(signal, sr)
    feat[:, f_beg : f_beg + f_wid] = extra_feat[:, f_beg : f_beg + f_wid]
    return feat


def spec_erase(self, feat):
    feat = torch.from_numpy(feat.reshape(1, 32, 84))
    feat = self.rand_erase(feat)
    feat = feat.reshape(32, 84).numpy()
    return feat

    f_wid = randint(0, 10)
    f_beg = randint(0, 84 - f_wid)
    t_wid = randint(0, 14)
    t_beg = randint(0, 32 - t_wid)
    feat[t_beg : t_beg + t_wid, f_beg : f_beg + f_wid] = 0
    return feat


def main():
    wav_path = ""  # undone
    try:
        y, sr = soundfile.read()
    except:
        raise AssertionError("Unsupported file: %s" % wav_path)


if __init__ == "__main__":
    main()
