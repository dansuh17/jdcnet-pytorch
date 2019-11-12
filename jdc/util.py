from typing import Union
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
    """
    Load an audio file and convert to log magnitude spectrogram.

    Args:
        audio_path(str): audio file path
        target_sr(int): target sample rate
        n_fft(int): number of FFT bins
        hop_length(int): window hop size in samples
        win_length(int): window size
        eps(float): epsilon value to add to zero energy bins so that log-magnitude spectrogram
            is numerically stable

    Returns:
        log magnitude spectrogram(np.ndarray): shape = (freq_bins, num_frames)
    """
    # load and resample
    y, sr = librosa.load(audio_path)
    y_resamp = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    spec = librosa.stft(y_resamp, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag_spec = np.abs(spec)

    # set epsilon where value is 0 - which will cause NaN in the log-magnitude spectrogram
    mag_spec[mag_spec == 0] = eps
    log_mag = np.log(mag_spec)
    return log_mag


def midi_to_frequency(midi_num: Union[int, np.ndarray]):
    midi_A4 = 69
    hz_A4 = 440
    x = np.exp(np.log(2) / 12)  # conversion
    return hz_A4 * (x ** (midi_num - midi_A4))


class PitchLabeler:
    def __init__(self, low=38, high=83, num_pitch_labels=721, nonvoice_threshold=0.1):
        """
        Class that helps labeling pitches. Provided a frequency, the `label()` method
        labels the frequency into one of the quantized frequency classes.
        The number of labels should be provided as `num_pitch_labels`, and
        the frequency range from `low` to `high` is divided equally in the log scale.
        Total number of labels includes 'nonvoice' label, which is determined by the
        `nonvoice_threshold` value.

        default: 45 semitones from D3 to B6 = 721 + nonvoice label = 722 labels

        Args:
            low(int): MIDI number for lower pitch (default: 38 (D2))
            high(int): MIDI number for higher pitch (default: 83 (B5))
            num_pitch_labels(int): number of pitch labels
        """
        self.num_pitch_labels = num_pitch_labels
        self.num_labels = num_pitch_labels + 1  # includes 'nonvoice' label
        self.nonvoice_threshold = nonvoice_threshold
        self.label_nonvoice = self.num_labels - 1
        self.label_midis = np.linspace(low, high, num=num_pitch_labels)
        self.label_hz = midi_to_frequency(self.label_midis)

    def label(self, freq: float):
        """
        Given a frequency, return the pitch label.

        Args:
            freq(float): frequency

        Returns:
            pitch label
        """
        if freq < self.nonvoice_threshold:
            return self.label_nonvoice
        else:
            return self.interval_index(freq)

    def midi(self, label: int, round_int=True) -> Union[int, float]:
        """
        Convert a label value to MIDI number.

        Args:
            label(int): pitch label - must be in range [0, self.num_pitch_labels - 1]
            round_int(bool): if set to true, it rounds to nearest MIDI integer

        Returns:
            midi_val: midi value
        """
        if label == self.label_nonvoice:
            return -1

        midi_val = self.label_midis[label]
        if round_int:
            midi_val = round(midi_val)
        return midi_val

    def interval_index(self, freq: float):
        idx = self.num_pitch_labels // 2
        start = 0
        end = self.num_pitch_labels - 1

        # binary search
        while True:
            if self.label_hz[idx] < freq:
                if idx == end:
                    return end
                elif freq < self.label_hz[idx + 1]:
                    return idx
                else:
                    start = idx + 1
                    idx = (start + end) // 2
            else:
                if idx == start:
                    return start
                elif freq > self.label_hz[idx - 1]:
                    return idx
                else:
                    end = idx - 1
                    idx = (start + end) // 2
