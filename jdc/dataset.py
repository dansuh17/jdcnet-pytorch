from collections import namedtuple
import json
import os

import librosa
import numpy as np
import torch.utils.data.dataset as dataset

# represents metadata for single source (mix)
SourceInfo = namedtuple(
    'SourceInfo',
    ['fname', 'audio_path', 'melody1_path', 'melody2_path', 'melody3_path',
     'artist', 'title', 'genre', 'is_excerpt', 'is_instrumental', 'n_sources'])


def medleydb_preprocess(root: str, target_sr=8000):
    for f in os.listdir(root):
        fpath = os.path.join(root, f)
        print(fpath)
        y, sr = librosa.load(fpath)

        # resample
        y_resamp = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

        mag_spec = librosa.stft(y_resamp, n_fft=1024, hop_length=80, win_length=1024)
        log_mag = np.log(mag_spec)
        print(log_mag.shape)


class MedleyDBMelodyDataset(dataset.Dataset):
    def __init__(self, metadata_path: str):
        with open(metadata_path, 'r') as meta_f:
            metadata = json.load(meta_f)

        self.source_infos = []
        for fname, value in metadata.items():
            sinfo = SourceInfo(
                fname=fname,
                audio_path=value['audio_path'],
                melody1_path=value['melody1_path'],
                melody2_path=value['melody2_path'],
                melody3_path=value['melody3_path'],
                artist=value['artist'],
                title=value['title'],
                genre=value['genre'],
                is_excerpt=value['is_excerpt'],
                is_instrumental=value['is_instrumental'],
                n_sources=value['n_sources'],
            )
            self.source_infos.append(sinfo)

    def __len__(self):
        # TODO: not right
        return len(self.source_infos)

    def __getitem__(self, i):
        # TODO: not right
        return self.source_infos[i]


if __name__ == '__main__':
    db = MedleyDBMelodyDataset('/Users/dansuh/datasets/MedleyDB-Melody/medleydb_melody_metadata.json')
    medleydb_preprocess('/Users/dansuh/datasets/MedleyDB-Melody/audio')
