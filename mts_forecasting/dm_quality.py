import os
import torch
import numpy as np
import lightning.pytorch as pl

from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from mts_forecasting.process_quality import process_data
from sklearn.model_selection import train_test_split


class LitDataModule(pl.LightningDataModule):
    train_inp = None
    train_trg = None
    test_inp = None
    test_trg = None

    def __init__(self, args: ArgumentParser, config: dict):
        super().__init__()
        self.seed = config["data_seed"]
        self.batch_size = config["batch_size"]
        self.segment_len = config["segment_len"]
        self.seq_len = config["sequence_len"]
        self.sample_time = config["sample_time"]

        self.data = process_data(root=config["root"], download=True)

        self.n_workers = args.n_workers
        self.pin_memory = args.pin_memory
        self.persistent = args.persistent_workers

        self.input_std = None
        self.trg_std = None
        self.train_test_split()

    def train_test_split(self):
        # all columns that contain "GT"
        trg_cols = [c for c in self.data.columns if "GT" in c]
        trg_indices = [self.data.columns.get_loc(c) for c in trg_cols]

        # all columns that do not contain "GT" nor "Date" nor "Time"
        inp_cols = [c for c in self.data.columns if "GT" not in c and "Date" not in c and "Time" not in c]
        inp_indices = [self.data.columns.get_loc(c) for c in inp_cols]

        assert self.seq_len <= self.segment_len, "Sequence length cannot be longer than the data time due to leakage"

        data = self.data.to_numpy()

        processed_data = self.process_data(data, self.segment_len, self.seq_len, trg_indices, inp_indices, self.seed)
        self.train_inp = processed_data['train_input']
        self.train_trg = processed_data['train_target']
        self.test_inp = processed_data['test_input']
        self.test_trg = processed_data['test_target']

    @staticmethod
    def process_data(data, l, n, trg_cols_indices, inp_cols_indices, seed=0):
        """
        Process the data with a specific RNG for the splitting process.
        """

        # Step 1: Split data into l-length segments
        segments = [data[i:i + l] for i in range(0, len(data), l) if i + l <= len(data)]

        # Step 2: Randomly split segments into train and test sets using the RNG
        train_segments, test_segments = train_test_split(segments, test_size=0.2, random_state=seed)

        # Step 3: Sub-segment the data into n-length samples with overlap
        def create_subsegments(segments, n):
            subsegments = []
            for segment in segments:
                subsegments.extend([segment[i:i + n] for i in range(len(segment) - n + 1)])
            return subsegments

        train_samples = create_subsegments(train_segments, n)
        test_samples = create_subsegments(test_segments, n)

        # Step 4: Calculate mean and std on train set
        train_data = np.vstack(train_samples)
        mean = np.mean(train_data, axis=0)
        std = np.std(train_data, axis=0)

        # Step 5: Standardize all data
        standardize = lambda x: (x - mean) / std
        train_data_standardized = [standardize(sample) for sample in train_samples]
        test_data_standardized = [standardize(sample) for sample in test_samples]

        # Separate input and target data
        train_inp = [seg[:, inp_cols_indices] for seg in train_data_standardized]
        train_trg = [seg[:, trg_cols_indices] for seg in train_data_standardized]
        test_inp = [seg[:, inp_cols_indices] for seg in test_data_standardized]
        test_trg = [seg[:, trg_cols_indices] for seg in test_data_standardized]

        train_input_tensor = torch.tensor(np.stack(train_inp, axis=0)).float()
        train_target_tensor = torch.tensor(np.stack(train_trg, axis=0)).float()
        test_input_tensor = torch.tensor(np.stack(test_inp, axis=0)).float()
        test_target_tensor = torch.tensor(np.stack(test_trg, axis=0)).float()

        return {
            'train_input': train_input_tensor,
            'train_target': train_target_tensor,
            'test_input': test_input_tensor,
            'test_target': test_target_tensor
        }

    def train_dataloader(self):
        dataset = MTSDataset(self.train_inp, self.train_trg)
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.n_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent,
                          collate_fn=MTSDataset.collate_fn)

    def val_dataloader(self):
        dataset = MTSDataset(self.test_inp, self.test_trg)
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.n_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent,
                          collate_fn=MTSDataset.collate_fn)

    def test_dataloader(self):
        dataset = MTSDataset(self.test_inp, self.test_trg)
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.n_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent,
                          collate_fn=MTSDataset.collate_fn)


class MTSDataset(Dataset):
    def __init__(self, inp_data, trg_data):
        self.input = inp_data
        self.trg_data = trg_data
        self.inp_idx = torch.arange(len(inp_data))
        print(inp_data.shape)

    @staticmethod
    def collate_fn(batch):
        x = torch.stack([b[0] for b in batch], dim=1)
        y = torch.stack([b[1] for b in batch], dim=1)
        return x, y

    def __len__(self):
        return len(self.inp_idx)

    def __getitem__(self, idx):
        s = self.inp_idx[idx]
        x = self.input[s]
        y = self.trg_data[s]
        return x, y