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
SpecHz = namedtuple('SpecHz', ['name', 'spec', 'hz', 'isvoice', 'start_idx'])


def medleydb_preprocess(source_info: SourceInfo, out_root: str, target_sr=8000):
    os.makedirs(out_root, exist_ok=True)

    df_melody = pd.read_csv(source_info.melody1_path, header=None, names=['sec', 'hz'])
    df_melody = df_melody[df_melody['hz'] > 0.1]  # filter out non-melodic regions

    # load and resample
    y, sr = librosa.load(source_info.audio_path)
    y_resamp = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    mag_spec = librosa.stft(y_resamp, n_fft=1024, hop_length=80, win_length=1024)
    log_mag = np.log(mag_spec)

    num_frames = log_mag.shape[1]
    chunk_size = 31
    for start_idx in range(0, num_frames, chunk_size):
        end_idx = start_idx + chunk_size
        chunk = log_mag[:, start_idx:end_idx]
        chunk_length = chunk.shape[1]
        if chunk_length != chunk_size:
            continue

        start_time, end_time = librosa.core.frames_to_time(
            (start_idx, end_idx), sr=target_sr, hop_length=80, n_fft=1024)

        melody_chunk = df_melody[df_melody['sec'].between(start_time, end_time)]

        is_voice = 1
        if len(melody_chunk) == 0:
            is_voice = 0  # indicate that no melody exists in this spectrogram chunk

        hz = melody_chunk['hz'].mean()
        print(start_idx, start_time, end_time)
        print(melody_chunk)

        out_path = os.path.join(out_root, f'{source_info.fname}_{start_idx}.pkl')
        with open(out_path, 'wb') as wf:
            spec_hz = SpecHz(
                name=source_info.fname,
                spec=chunk,
                hz=hz,
                isvoice=is_voice,
                start_idx=start_idx)
            pickle.dump(spec_hz, wf)


def read_source_infos(root: str, metadata_path: str):
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
    source_infos = read_source_infos(in_root, metadata_path)
    for si in source_infos:
        medleydb_preprocess(si, out_root)


class MedleyDBMelodyDataset(dataset.Dataset):
    def __init__(self, root: str):
        self.datas = []
        for fname in os.listdir(root):
            with open(os.path.join(root, fname)) as f:
                spec_hz: SpecHz = pickle.load(f)
                self.datas.append(spec_hz)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, i):
        return self.datas[i]


if __name__ == '__main__':
    t = librosa.core.frames_to_time(19066, sr=8000, hop_length=80, n_fft=1024)
    print(t)
    # db = MedleyDBMelodyDataset('/Users/dansuh/datasets/', 'MedleyDB-Melody/medleydb_melody_metadata.json')
    # medleydb_preprocess('/Users/dansuh/datasets/MedleyDB-Melody/audio')
    preprocess('/Users/dansuh/datasets/', './out_root', 'MedleyDB-Melody/medleydb_melody_metadata.json')


