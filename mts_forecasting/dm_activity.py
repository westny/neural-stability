# Copyright 2024, Theodor Westny. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import numpy as np
import pandas as pd
import lightning.pytorch as pl

from argparse import Namespace
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from mts_forecasting.dataset import MTSDataset
from mts_forecasting.process_activity import process_activity_data


class LitDataModule(pl.LightningDataModule):
    def __init__(self,
                 args: Namespace,
                 config: dict
                 ) -> None:
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
            process_activity_data(config["root"])
        df = pd.read_csv(path)

        processed_data = self.process_data(df, self.seq_len, self.seed)

        self.train = MTSDataset(processed_data["train_input"], processed_data["train_target"])
        self.val = MTSDataset(processed_data["val_input"], processed_data["val_target"])
        self.test = MTSDataset(processed_data["test_input"], processed_data["test_target"])

    @staticmethod
    def process_data(df: pd.DataFrame,
                     seq_len: int,
                     seed: int = 0) -> dict:
        """
        Process the data with a specific RNG for the splitting process.
        """

        # note that these are the indices after dropping the record_id column
        input_idx = [0, 1, 2, 3, 4, 5, 9, 10, 11]  # Both ankles + belt
        output_idx = [6, 7, 8]  # chest

        # get all unique record_ids
        record_ids = df.record_id.unique()

        train_ids, test_ids = train_test_split(record_ids, test_size=0.2, random_state=seed)
        train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=seed)

        train_df = df[df.record_id.isin(train_ids)]
        val_df = df[df.record_id.isin(val_ids)]
        test_df = df[df.record_id.isin(test_ids)]

        # breakpoint()

        def create_segments(df, r_ids):
            indices = []
            for r_id in r_ids:
                indices += df.index[df.record_id == r_id].tolist()[:-seq_len]

            # Drop the 'record_id' column
            df_dropped = df.drop(columns=["record_id"])

            def get_single_segment(idx):
                segment = df_dropped.loc[idx:idx + seq_len - 1].to_numpy()
                return segment, len(segment)

            segments = [segment for idx in indices for segment, ln in [get_single_segment(idx)] if ln == seq_len]

            stacked_data = np.stack(segments)
            inp = torch.from_numpy(stacked_data[..., input_idx]).float()
            out = torch.from_numpy(stacked_data[..., output_idx]).float()
            return inp, out

        train_inp, train_out = create_segments(train_df, train_ids)
        val_inp, val_out = create_segments(val_df, val_ids)
        test_inp, test_out = create_segments(test_df, test_ids)

        return {
            "train_input": train_inp,
            "train_target": train_out,
            "val_input": val_inp,
            "val_target": val_out,
            "test_input": test_inp,
            "test_target": test_out
        }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train,
                          shuffle=True,
                          collate_fn=MTSDataset.collate_fn,
                          batch_size=self.batch_size,
                          num_workers=self.n_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers,
                          )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val if not self.evaluate else self.test,
                          collate_fn=MTSDataset.collate_fn,
                          batch_size=self.batch_size,
                          num_workers=self.n_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers
                          )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test,
                          collate_fn=MTSDataset.collate_fn,
                          batch_size=self.batch_size,
                          num_workers=self.n_workers)
