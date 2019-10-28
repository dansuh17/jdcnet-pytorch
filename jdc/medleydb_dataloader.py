import random
import math
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchland.datasets.loader_maker import DataLoaderBuilder
from .dataset import MedleyDBMelodyDataset


class MedleyDBDataLoaderBuilder(DataLoaderBuilder):
    def __init__(self, data_root: str, batch_size: int, num_workers=8, train_ratio=0.8, val_ratio=0.1):
        super().__init__()
        self.dataset = MedleyDBMelodyDataset(root=data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers

        num_data = len(self.dataset)
        indices = list(range(num_data))
        random.shuffle(indices)
        num_train = math.floor(num_data * train_ratio)
        num_val = math.floor(num_data * val_ratio)
        self.train_idx, valtest_idx = indices[:num_train], indices[num_train:]
        self.val_idx, self.test_idx = valtest_idx[:num_val], valtest_idx[num_val:]

    def make_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            sampler=SubsetRandomSampler(self.train_idx),
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def make_validate_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            sampler=SubsetRandomSampler(self.val_idx),
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def make_test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            sampler=SubsetRandomSampler(self.test_idx),
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )
