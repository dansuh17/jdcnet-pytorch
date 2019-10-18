from collections import namedtuple
import json
import os
import pickle

import librosa
import numpy as np
import pandas as pd
import torch.utils.data.dataset as dataset

# represents metadata for single source (mix)
SourceInfo = namedtuple(
    'SourceInfo',
    ['fname', 'audio_path', 'melody1_path', 'melody2_path', 'melody3_path',
     'artist', 'title', 'genre', 'is_excerpt', 'is_instrumental', 'n_sources'])

# represents data tuple representing spectrogram
# chunk and the melody note it's representing
SpecHz = namedtuple('SpecHz', ['name', 'spec', 'label', 'isvoice', 'start_idx'])


def midi_to_frequency(midi_num):
    midi_A4 = 69
    hz_A4 = 440
    x = np.exp(np.log(2) / 12)  # conversion # multiplier
    return hz_A4 * (x ** (midi_num - midi_A4))


class PitchLabeler:
    def __init__(self, low=38, high=83, num_pitch_labels=721, nonvoice_threshold=0.1):
        """

        Args:
            low:  D2
            high:  B5
            num_pitch_labels:
        """
        """
        45 semitones from D3 to B6 = 721 + nonvoice label = 722 labels
        """
        self.num_pitch_labels = num_pitch_labels
        self.num_labels = num_pitch_labels + 1  # includes 'nonvoice' label
        self.nonvoice_threshold = nonvoice_threshold
        self.label_nonvoice = self.num_labels - 1
        self.label_midis = np.linspace(low, high, num=num_pitch_labels)
        self.label_hz = midi_to_frequency(self.label_midis)

    def label(self, freq: float):
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


def get_melody_labels_by_frame(
        source_info: SourceInfo, pitch_labeler: PitchLabeler,
        num_frames: int, frames_to_time):
    # read melody info
    df_melody = pd.read_csv(source_info.melody1_path, header=None, names=['sec', 'hz'])

    frame_indices = np.arange(num_frames)
    frame_times = frames_to_time(frame_indices)

    labels = []
    isvoice = []
    for start_idx in range(num_frames):
        start_time = frame_times[start_idx]
        if start_idx + 1 >= num_frames:
            melody_part = df_melody[df_melody['sec'] > start_time]
        else:
            end_time = frame_times[start_idx + 1]
            melody_part = df_melody[df_melody['sec'].between(start_time, end_time)]

        melody_part_pitched = melody_part[melody_part['hz'] > 0.1]

        if len(melody_part_pitched) == 0:
            label = pitch_labeler.label_nonvoice
            voice = False
        else:
            label = pitch_labeler.label(melody_part_pitched['hz'].mean())
            voice = True

        labels.append(label)
        isvoice.append(1 if voice else 0)
    return labels, isvoice


def medleydb_preprocess(
        source_info: SourceInfo, out_root: str,
        pitch_labeler: PitchLabeler, target_sr=8000):
    """
    Preprocess a single source of MedleyDB mix.

    Args:
        source_info(SourceInfo): source audio information
        out_root(str): output directory to save the resulting file
        target_sr(int): target sample rate
        pitch_labeler(PitchLabeler): pitch labeler
    """
    chunk_size = 31

    os.makedirs(out_root, exist_ok=True)

    # load and resample
    y, sr = librosa.load(source_info.audio_path)
    y_resamp = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    mag_spec = librosa.stft(y_resamp, n_fft=1024, hop_length=80, win_length=1024)
    log_mag = np.log(np.abs(mag_spec))
    num_frames = log_mag.shape[1]

    frames_to_time = lambda frame_indices: librosa.core.frames_to_time(
        frame_indices, sr=target_sr, hop_length=80, n_fft=1024)

    # TODO: provide only the times
    print('extracting melody...')
    pitch_labels, isvoice = get_melody_labels_by_frame(
        source_info, pitch_labeler, num_frames, frames_to_time)

    assert(len(pitch_labels) == num_frames)
    assert(len(isvoice) == num_frames)

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
        out_path = os.path.join(out_root, f'{source_info.fname}_{start_idx}.pkl')
        with open(out_path, 'wb') as wf:
            spec_hz = SpecHz(
                name=source_info.fname,
                # swap: (freq_bin, chunk_size) => (chunk_size, freq_bin)
                spec=np.swapaxes(chunk, axis1=0, axis2=1),
                label=pitch_labels_chunk,
                isvoice=isvoice_chunk,
                start_idx=start_idx)
            pickle.dump(spec_hz, wf)


def read_source_infos(root: str, metadata_path: str):
    """
    Read MedleyDB data items from

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
    for si in source_infos:
        print(f'Preprocessing: {si.fname}')
        medleydb_preprocess(si, out_root, pitch_labeler)


class MedleyDBMelodyDataset(dataset.Dataset):
    def __init__(self, root: str):
        self.fnames = [
            os.path.join(root, fname)
            for fname in os.listdir(root)]

    @staticmethod
    def _read_data(path: str):
        with open(path, 'rb') as f:
            spec_hz: SpecHz = pickle.load(f)
        return spec_hz.spec, spec_hz.label, spec_hz.isvoice

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, i):
        return self._read_data(self.fnames[i])


if __name__ == '__main__':
    # t = librosa.core.frames_to_time(19066, sr=8000, hop_length=80, n_fft=1024)
    # print(t)

    # db = MedleyDBMelodyDataset('/Users/dansuh/datasets/', 'MedleyDB-Melody/medleydb_melody_metadata.json')
    # medleydb_preprocess('/Users/dansuh/datasets/MedleyDB-Melody/audio')
    preprocess('/Users/dansuh/datasets/', './out_root', 'MedleyDB-Melody/medleydb_melody_metadata.json')
