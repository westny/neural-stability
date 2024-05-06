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
from argparse import Namespace
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from latent_dynamics.dataset import CustomHDF5Dataset


class LitDataModule(pl.LightningDataModule):
    def __init__(self,
                 args: Namespace,
                 config: dict) -> None:
        super().__init__()
        self.batch_size = config["batch_size"]
        self.dataset = config["name"]
        self.root = config["root"]

        self.inp_len = config["inp_len"]
        self.trg_len = config["trg_len"]
        self.test_trg_len = config["test_trg_len"]
        self.sample_time = config["sample_time"]

        self.n_workers = args.n_workers
        self.pin_memory = args.pin_memory
        self.persistent_workers = args.persistent_workers

        assert self.sample_time % 0.05 == 0, "Step size must be a multiple of the original sampling time 0.05"
        assert self.sample_time >= 0.05, "Step size must be greater than or equal to the original sampling time 0.05"

        self.data_path = os.path.join(self.root, self.dataset)
        # check if /data/args.dataset exists, else raise error
        if not os.path.exists(self.data_path):
            raise ValueError(f"Could not find data. Please run tf_converter_hdf5.py to download the data.")

        self.train = CustomHDF5Dataset(self.data_path, "train",
                                       self.inp_len, self.trg_len, self.sample_time)
        self.val = CustomHDF5Dataset(self.data_path, "val",
                                     self.inp_len, self.trg_len, self.sample_time)
        self.test = CustomHDF5Dataset(self.data_path, "test",
                                      self.inp_len, self.test_trg_len, self.sample_time)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train,
                          shuffle=True,
                          batch_size=self.batch_size,
                          num_workers=self.n_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers,
                          prefetch_factor=4,
                          )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          num_workers=self.n_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers
                          )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.n_workers)