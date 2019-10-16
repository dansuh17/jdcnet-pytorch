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
SpecHz = namedtuple('SpecHz', ['name', 'spec', 'hz', 'label', 'isvoice', 'start_idx'])


def midi_to_frequency(midi_num):
    midi_A4 = 69
    hz_A4 = 440
    x = np.exp(np.log(2) / 12)  # conversion # multiplier
    return hz_A4 * (x ** (midi_num - midi_A4))


def interval_index(val, intervals):
    l = len(intervals)
    idx = l // 2
    start = 0
    end = l - 1
    while True:
        if intervals[idx] < val:
            if idx == end:
                return end
            elif val < intervals[idx + 1]:
                return idx
            else:
                start = idx + 1
                idx = (start + end) // 2
        else:
            if idx == start:
                return start
            elif val > intervals[idx - 1]:
                return idx
            else:
                end = idx - 1
                idx = (start + end) // 2


def medleydb_preprocess(source_info: SourceInfo, out_root: str, target_sr=8000):
    """
    Preprocess a single source of MedleyDB mix.

    Args:
        source_info(SourceInfo): source audio information
        out_root(str): output directory to save the resulting file
        target_sr(int): target sample rate
    """
    os.makedirs(out_root, exist_ok=True)

    df_melody = pd.read_csv(source_info.melody1_path, header=None, names=['sec', 'hz'])
    df_melody = df_melody[df_melody['hz'] > 0.1]  # filter out non-melodic regions

    # load and resample
    y, sr = librosa.load(source_info.audio_path)
    y_resamp = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    mag_spec = librosa.stft(y_resamp, n_fft=1024, hop_length=80, win_length=1024)
    log_mag = np.log(np.abs(mag_spec))

    # TODO: factor out
    # for quantizing frequency
    """
    45 semitones from D3 to B6 = 721 + nonvoice label = 722 labels
    """
    low = 38  # D2
    high = 83  # B5
    num_pitch_labels = 721
    num_labels = num_pitch_labels + 1  # includes 'nonvoice' label
    label_nonvoice = num_labels - 1
    label_midis = np.linspace(low, high, num=num_pitch_labels)
    label_hz = midi_to_frequency(label_midis)

    # split the spectrogram in chunks
    num_frames = log_mag.shape[1]
    chunk_size = 31
    for start_idx in range(0, num_frames, chunk_size):
        end_idx = start_idx + chunk_size
        chunk = log_mag[:, start_idx:end_idx]

        chunk_length = chunk.shape[1]
        # most probably the chunk is short on frames at the end of audio
        if chunk_length != chunk_size:
            continue

        # convert frame position to time range
        start_time, end_time = librosa.core.frames_to_time(
            (start_idx, end_idx), sr=target_sr, hop_length=80, n_fft=1024)

        melody_chunk = df_melody[df_melody['sec'].between(start_time, end_time)]

        is_voice = 1
        if len(melody_chunk) == 0:
            is_voice = 0  # indicate that no melody exists in this spectrogram chunk

        hz = melody_chunk['hz'].mean()

        if is_voice == 1:
            label = interval_index(hz, label_hz)  # find the interval index == label
        else:
            label = label_nonvoice

        print(hz, label)

        # save the processed data to pickle file
        out_path = os.path.join(out_root, f'{source_info.fname}_{start_idx}.pkl')
        with open(out_path, 'wb') as wf:
            spec_hz = SpecHz(
                name=source_info.fname,
                spec=np.swapaxes(chunk, 0, 1),  # swap: (freq_bin, chunk_size) => (chunk_size, freq_bin)
                hz=hz,
                label=label,
                isvoice=is_voice,
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
    for si in source_infos:
        medleydb_preprocess(si, out_root)


class MedleyDBMelodyDataset(dataset.Dataset):
    def __init__(self, root: str):
        self.fnames = [
            os.path.join(root, fname)
            for fname in os.listdir(root)]
        print(self.fnames)

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

    # print(interval_index(812, [0, 1, 2, 4, 5]))
