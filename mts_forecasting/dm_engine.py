import os
import torch
import numpy as np
import lightning.pytorch as pl

from argparse import ArgumentParser
from numpy.random import default_rng
from torch.utils.data import Dataset, DataLoader
from mts_forecasting.process_engine import process_data


class LitDataModule(pl.LightningDataModule):
    def __init__(self, args: ArgumentParser, config: dict):
        super().__init__()
        self.seed = config["data_seed"]
        self.batch_size = config["batch_size"]
        self.segment_len = config["segment_len"]
        self.seq_len = config["sequence_len"]
        self.sample_time = config["sample_time"]

        self.data, _, self.residual = process_data(root=config["root"], download=True)

        self.n_workers = args.n_workers
        self.pin_memory = args.pin_memory
        self.persistent = args.persistent_workers

        self.input_std = None
        self.trg_std = None
        self.inp_data, self.trg, self.tr_idx, self.te_idx, self.input_std, self.trg_std = self.train_test_split()

    def train_test_split(self, split=0.8, oversample=1):
        ms = self.residual.measurements
        meas_data = torch.stack([self.data[m] for m in ms], dim=-1)
        trg_data = self.data[self.residual.sensor].unsqueeze(-1)
        assert self.seq_len <= self.segment_len, "Sequence length cannot be longer than the data time due to leakage"

        start_idx = np.arange(0, meas_data.shape[0] - self.segment_len, self.segment_len)
        n_train_indices = int(len(start_idx) * split)

        rng = default_rng(self.seed)

        train_start_indices = rng.choice(start_idx, n_train_indices, replace=False)
        test_start_indices = start_idx[(np.isin(start_idx, train_start_indices) == False)]

        # standardize input data
        input_standardization = {}

        for mi, m_name in enumerate(ms):
            ys = []
            curr_meas = meas_data[:, mi]
            for i in train_start_indices:
                y = curr_meas[i:i + self.segment_len]
                ys.append(y)
            ys = torch.cat(ys, dim=0)
            mu = ys.mean()
            std = ys.std()
            meas_data[:, mi] = (curr_meas - mu) / std
            input_standardization[m_name] = (mu, std)

        # standardize trg data
        trg_standardization = {}
        trg = []
        for i in train_start_indices:
            trg.append(trg_data[i:i + self.segment_len])
        trg = torch.cat(trg, dim=0)
        trg_mu = trg.mean()
        trg_std = trg.std()
        trg_standardization[self.residual.sensor] = (trg_mu, trg_std)

        trg_data = (trg_data - trg_mu) / trg_std

        train_start_indices = torch.tensor(train_start_indices)
        test_start_indices = torch.tensor(test_start_indices)

        train_extended = []
        test_extended = []

        # Oversample data
        for i in range(0, self.segment_len + 1, oversample):
            if i + self.seq_len > self.segment_len:
                break
            tri = train_start_indices + i
            tei = test_start_indices + i

            train_extended.append(tri)
            test_extended.append(tei)

        train_start_indices, _ = torch.sort(torch.cat(train_extended))
        test_start_indices, _ = torch.sort(torch.cat(test_extended))

        # get overlapping points
        final = []
        for ti in (train_start_indices, test_start_indices):
            overlap = []
            for i in range(len(ti) - 1):
                ni = ti[i]
                nii = ti[i + 1]
                if ni == nii:
                    continue
                if nii > ni:
                    if nii - ni == self.seq_len:
                        add = torch.arange(ni, nii, oversample)
                        overlap.append(add)
            try:
                overlap = torch.cat(overlap)
                ti = torch.cat((ti, overlap))
            except RuntimeError:
                pass
            final.append(ti)

        train_start_indices, test_start_indices = final

        # final sort
        train_start_indices, _ = torch.sort(train_start_indices)
        test_start_indices, _ = torch.sort(test_start_indices)

        return meas_data, trg_data, train_start_indices, test_start_indices, input_standardization, trg_standardization

    def train_dataloader(self):
        dataset = AfterTreatDataset(self.inp_data, self.trg, self.tr_idx, self.sample_time, self.seq_len)
        dataset_ext = AfterTreatDataset(self.inp_data, self.trg, self.tr_idx, self.sample_time, self.seq_len)
        data_loader = DataLoader(dataset,
                                 batch_size=self.batch_size,
                                 shuffle=True,
                                 num_workers=self.n_workers,
                                 pin_memory=self.pin_memory,
                                 persistent_workers=self.persistent,
                                 collate_fn=AfterTreatDataset.collate_fn)
        data_loader_extended = DataLoader(dataset_ext, batch_size=self.batch_size, shuffle=True,
                                          num_workers=self.n_workers, pin_memory=self.pin_memory,
                                          collate_fn=AfterTreatDataset.collate_fn)

        return {'a': data_loader, 'b': data_loader_extended}

    def val_dataloader(self):
        dataset = AfterTreatDataset(self.inp_data, self.trg, self.te_idx, self.sample_time, self.seq_len)
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.n_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent,
                          collate_fn=AfterTreatDataset.collate_fn)

    def test_dataloader(self):
        dataset = AfterTreatDataset(self.inp_data, self.trg, self.te_idx, self.sample_time, self.seq_len)
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.n_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent,
                          collate_fn=AfterTreatDataset.collate_fn)


class AfterTreatDataset(Dataset):
    def __init__(self, meas_data, trg_data, start_index, sampling_time, seq_len=100):
        self.meas_data = meas_data
        self.trg_data = trg_data
        self.inp_idx = start_index
        self.b_time = seq_len
        self.d_meas_data = torch.diff(self.meas_data, dim=0) / sampling_time
        # self.input = torch.cat((self.meas_data[:-1], self.d_meas_data), dim=-1)
        self.input = self.meas_data

    @staticmethod
    def collate_fn(batch):
        x = torch.stack([b[0] for b in batch], dim=1)
        y = torch.stack([b[1] for b in batch], dim=1)
        return x, y

    def __len__(self):
        return len(self.inp_idx)

    def __getitem__(self, idx):
        s = self.inp_idx[idx]
        y = self.trg_data[s:s + self.b_time]
        x = self.input[s:s + self.b_time]
        return x, y
