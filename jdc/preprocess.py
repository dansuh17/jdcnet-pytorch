from collections import namedtuple
import os
import json
import pickle
import librosa
import numpy as np
import pandas as pd
from .dataset import SpecHz
from .util import log_magnitude_spectrogram


# represents metadata for single source (mix) - used for preprocessing MedleyDB dataset
SourceInfo = namedtuple(
    'SourceInfo',
    ['fname', 'audio_path', 'melody1_path', 'melody2_path', 'melody3_path',
     'artist', 'title', 'genre', 'is_excerpt', 'is_instrumental', 'n_sources'])


def midi_to_frequency(midi_num):
    midi_A4 = 69
    hz_A4 = 440
    x = np.exp(np.log(2) / 12)  # conversion # multiplier
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
            high(int): MIDI number for higher pitch (default: B5 (B5))
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


def medleydb_preprocess(
        source_info: SourceInfo, pitch_labeler: PitchLabeler,
        chunk_size=31, target_sr=8000, n_fft=1024, hop_length=80, win_length=1024):
    """
    Preprocess a single source of MedleyDB mix and generates chunked data.

    Args:
        source_info(SourceInfo): source audio information
        pitch_labeler(PitchLabeler): pitch labeler
        chunk_size(int): number of frames to cut the spectrogram into
        target_sr(int): target sample rate
        n_fft(int): number off fft bins
        hop_length(int): hop length for windowing in stft
        win_length(int): window length for stft

    Returns:
        spec_hz(SpecHz): includes chunked spectrogram, labeled pitches, and metadata
    """
    # load audio and convert to log-magnitude spectrogram
    log_mag = log_magnitude_spectrogram(
        source_info.audio_path, target_sr, n_fft, hop_length, win_length)
    num_frames = log_mag.shape[1]

    # calculate the elapsed time for each frame index
    frame_indices = np.arange(num_frames)
    frame_times = librosa.core.frames_to_time(
        frame_indices, sr=target_sr, hop_length=hop_length, n_fft=n_fft)

    pitch_labels, isvoice = get_melody_labels_by_frame(
        source_info, pitch_labeler, frame_times)

    assert(len(pitch_labels) == num_frames)
    assert(len(isvoice) == num_frames)

    # check data sanity
    assert(not np.any(np.isnan(log_mag)))
    assert(np.all(np.isfinite(log_mag)))

    # split the spectrogram in chunks
    for start_idx in range(0, num_frames, chunk_size):
        end_idx = start_idx + chunk_size
        chunk = log_mag[:, start_idx:end_idx]
        pitch_labels_chunk = pitch_labels[start_idx: end_idx]
        isvoice_chunk = isvoice[start_idx: end_idx]

        # most probably the chunk is short on frames at the end of audio
        chunk_length = chunk.shape[1]
        if chunk_length != chunk_size:
            continue

        # save the processed data to pickle file
        yield SpecHz(
            name=source_info.fname,
            # swap: (freq_bin, chunk_size) => (chunk_size, freq_bin)
            spec=np.swapaxes(chunk, axis1=0, axis2=1),
            label=pitch_labels_chunk,
            isvoice=isvoice_chunk,
            start_idx=start_idx)


def get_melody_labels_by_frame(
        source_info: SourceInfo, pitch_labeler: PitchLabeler, frame_times):
    # read melody info
    df_melody = pd.read_csv(source_info.melody1_path, header=None, names=['sec', 'hz'])

    num_frames = len(frame_times)

    labels = []  # array of pitch labels (integer values)
    isvoice = []  # array of bools saying whethere there is a voice
    for start_idx in range(num_frames):
        # determine the start / end times and choose the melody in the interval
        start_time = frame_times[start_idx]
        if start_idx + 1 >= num_frames:
            melody_part = df_melody[df_melody['sec'] > start_time]
        else:
            end_time = frame_times[start_idx + 1]
            melody_part = df_melody[df_melody['sec'].between(start_time, end_time)]

        melody_part_pitched = melody_part[melody_part['hz'] > 0.1]

        # no pitched melody exists in this chunk
        if len(melody_part_pitched) == 0:
            label = pitch_labeler.label_nonvoice
            voice = False
        else:
            label = pitch_labeler.label(melody_part_pitched['hz'].mean())
            voice = True

        labels.append(label)
        isvoice.append(1 if voice else 0)
    return np.array(labels), np.array(isvoice)


def read_source_infos(root: str, metadata_path: str):
    """
    Read MedleyDB data items from dataset root directory.

    Args:
        root(str): data root
        metadata_path(str): path to metadata file from data root

    Returns:
        list of SourceInfos
    """
    with open(os.path.join(root, metadata_path), 'r') as meta_f:
        metadata = json.load(meta_f)

    source_infos = []
    for fname, value in metadata.items():
        sinfo = SourceInfo(
            fname=fname,
            audio_path=os.path.join(root, value['audio_path']),
            melody1_path=os.path.join(root, value['melody1_path']),
            melody2_path=os.path.join(root, value['melody2_path']),
            melody3_path=os.path.join(root, value['melody3_path']),
            artist=value['artist'],
            title=value['title'],
            genre=value['genre'],
            is_excerpt=value['is_excerpt'],
            is_instrumental=value['is_instrumental'],
            n_sources=value['n_sources'],
        )
        source_infos.append(sinfo)
    return source_infos


def preprocess(in_root: str, out_root: str, metadata_path: str):
    """
    Preprocess the entire data from MedleyDB dataset.

    Args:
        in_root(str): input data root directory
        out_root(str): output data root directory
        metadata_path(str): path from in_root to metadata file
    """
    source_infos = read_source_infos(in_root, metadata_path)
    pitch_labeler = PitchLabeler()
    num_sources = len(source_infos)

    count = 0
    for i, si in enumerate(source_infos):
        print(f'Preprocessing: {si.fname} ({i + 1} / {num_sources})')
        for spec_hz in medleydb_preprocess(si, pitch_labeler):
            count += 1
            out_path = os.path.join(out_root, f'{spec_hz.name}_{spec_hz.start_idx}.pkl')
            print(f'Saving: {out_path}, count: {count}')
            with open(out_path, 'wb') as wf:
                pickle.dump(spec_hz, wf)


if __name__ == '__main__':
    # db = MedleyDBMelodyDataset('/Users/dansuh/datasets/', 'MedleyDB-Melody/medleydb_melody_metadata.json')
    preprocess(in_root='/Users/dansuh/datasets/', out_root='./out_root',
               metadata_path='MedleyDB-Melody/medleydb_melody_metadata.json')
