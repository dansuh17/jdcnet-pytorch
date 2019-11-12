import torch
import librosa
import numpy as np


def empty_onehot(target: torch.Tensor, num_classes: int):
    # target_size = (batch, dim1, dim2, ...)
    # one_hot size = (batch, dim1, dim2, ..., num_classes)
    onehot_size = target.size() + (num_classes, )
    return torch.FloatTensor(*onehot_size).zero_()


def to_onehot(target: torch.Tensor, num_classes: int, src_onehot: torch.Tensor = None):
    if src_onehot is None:
        one_hot = empty_onehot(target, num_classes)
    else:
        one_hot = src_onehot

    last_dim = len(one_hot.size()) - 1

    # creates a one hot vector provided the target indices
    # and the Tensor that holds the one-hot vector
    with torch.no_grad():
        one_hot = one_hot.scatter_(
            dim=last_dim, index=torch.unsqueeze(target, dim=last_dim), value=1.0)
    return one_hot


def log_magnitude_spectrogram(audio_path: str, target_sr=8000, n_fft=1024,
                              hop_length=80, win_length=1024, eps=1e-10):
    # load and resample
    y, sr = librosa.load(audio_path)
    y_resamp = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    spec = librosa.stft(y_resamp, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag_spec = np.abs(spec)

    # set epsilon where value is 0 - which will cause NaN in the log-magnitude spectrogram
    mag_spec[mag_spec == 0] = eps
    log_mag = np.log(mag_spec)
    return log_mag


