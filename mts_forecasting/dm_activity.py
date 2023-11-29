import os
import torch
import pandas as pd
import lightning.pytorch as pl

from typing import List
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from mts_forecasting.process_activity import process_activity_data


class LitDataModule(pl.LightningDataModule):
    def __init__(self, args: ArgumentParser, config: dict):
        super().__init__()
        self.seed = config["data_seed"]
        self.batch_size = config["batch_size"]
        self.seq_len = config["sequence_len"]
        self.sample_time = config["sample_time"]
        self.evaluate = config["evaluate"]

        self.n_workers = args.n_workers
        self.pin_memory = args.pin_memory
        self.persistent_workers = args.persistent_workers

        path = os.path.join(config["root"], "data.csv")

        # check if the data.csv file exists
        if not os.path.exists(path):
            # check if folder exists, otherwise create it
            # if not os.path.exists(config["root"]):
            #     os.mkdir(config["root"])
            process_activity_data(config["root"])
        df = pd.read_csv(path)
        train_ids, val_ids, test_ids = self.process_data(df, self.seed)

        self.train = ActivityDataset(df, train_ids, self.seq_len)
        self.val = ActivityDataset(df, val_ids, self.seq_len)
        self.test = ActivityDataset(df, test_ids, self.seq_len)

    @staticmethod
    def process_data(df, seed=0):
        """
        Process the data with a specific RNG for the splitting process.
        """
        # get all unique record_ids
        record_ids = df.record_id.unique()

        # shuffle the ids based on the data seed
        rng = torch.Generator().manual_seed(seed)
        record_ids = record_ids[torch.randperm(len(record_ids), generator=rng)]

        # split the record_ids into train, val and test
        train_ids = record_ids[:int(len(record_ids) * 0.6)]
        val_ids = record_ids[int(len(record_ids) * 0.6):int(len(record_ids) * 0.8)]
        test_ids = record_ids[int(len(record_ids) * 0.8):]

        return train_ids, val_ids, test_ids

    def train_dataloader(self):
        return DataLoader(self.train,
                          shuffle=True,
                          collate_fn=ActivityDataset.collate_fn,
                          batch_size=self.batch_size,
                          num_workers=self.n_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers,
                          )

    def val_dataloader(self):
        return DataLoader(self.val if not self.evaluate else self.test,
                          collate_fn=ActivityDataset.collate_fn,
                          batch_size=self.batch_size,
                          num_workers=self.n_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers
                          )

    def test_dataloader(self):
        return DataLoader(self.test,
                          collate_fn=ActivityDataset.collate_fn,
                          batch_size=self.batch_size,
                          num_workers=self.n_workers)


class ActivityDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 r_ids: List[str],
                 inp_len: int = 100,
                 out_len: int = 20):

        self.record_ids = r_ids
        self.inp_len = inp_len
        self.out_len = out_len

        self.input_idx = [1, 2, 3, 4, 5, 6, 10, 11, 12]  # Ankles + belt
        self.output_idx = [7, 8, 9]  # chest

        # filter out the current record_ids
        self.df = df[df.record_id.isin(r_ids)]

        # Get the number of rows for each record_id
        self.n_rows = {record_id: len(self.df[self.df.record_id == record_id]) for record_id in self.record_ids}

        # Get all the possible starting indices for each record_id
        self.start_indices = {record_id: torch.arange(self.n_rows[record_id] - inp_len)
                              for record_id in self.record_ids}

        # map all the starting indices to a unique label for data indexing
        self.start_idx_to_label = {}
        j = 0
        for record_id in self.record_ids:
            for i in range(len(self.start_indices[record_id])):
                self.start_idx_to_label[j] = (record_id, i)
                j += 1

    @staticmethod
    def collate_fn(batch):
        x = torch.stack([b[0] for b in batch], dim=1)
        y = torch.stack([b[1] for b in batch], dim=1)
        return x, y

    def __len__(self):
        return len(self.start_idx_to_label)

    def __getitem__(self, idx):
        r_id, start_idx = self.start_idx_to_label[idx]

        # get x as all the columns x1, y1, y2, .... , x4, y4, z4
        df = self.df[self.df.record_id == r_id]

        x = df.iloc[start_idx:start_idx + self.inp_len].iloc[:, self.input_idx].values
        y = df.iloc[start_idx:start_idx + self.inp_len].iloc[:, self.output_idx].values

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        return x, y
