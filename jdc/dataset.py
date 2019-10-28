from collections import namedtuple
import os
import pickle
import torch.utils.data.dataset as dataset


# represents data tuple representing spectrogram
# chunk and the melody note it's representing
SpecHz = namedtuple('SpecHz', ['name', 'spec', 'label', 'isvoice', 'start_idx'])


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
