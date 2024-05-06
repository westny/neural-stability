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
import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class SequentialToTensor:
    def __call__(self, pic: np.ndarray) -> torch.Tensor:
        """
        Convert a numpy.ndarray with shape (S, H, W, C) to torch.FloatTensor with shape (S, C, H, W).
        Pixel values are scaled from [0, 255] to [0.0, 1.0].
        """
        tensor = torch.from_numpy(pic.transpose((0, 3, 1, 2)))

        return tensor.float().div(255)


class CustomHDF5Dataset(Dataset):
    def __init__(self,
                 hdf5_file_path: str = "data/twoXpendulum",
                 train_test: str = "train",
                 inp_len: int = 10,
                 trg_len: int = 60,
                 sample_time: float = 0.05
                 ) -> None:
        """
        Args:
            hdf5_file_path (string): Path to the HDF5 file.
            train_test (string): train, val or test.
            inp_len (int): Number of input steps.
            trg_len (int): Number of target steps.
            sample_time (float): Sampling time of the data.
        """
        # self.file_path = hdf5_file_path + train_test + ".hdf5"
        self.file_path = os.path.join(hdf5_file_path, train_test + ".hdf5")
        self.sample_time = sample_time
        self.slice = int(self.sample_time / 0.05)
        self.inp_len = inp_len * self.slice
        self.trg_len = trg_len * self.slice

        self.transform = transforms.Compose([
            SequentialToTensor(),
        ])

        with h5py.File(self.file_path, 'r') as file:
            # Data is stored under a dataset named 'images'
            self.length = file['images'].shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with h5py.File(self.file_path, 'r') as file:
            # Data is stored under a dataset named 'images'
            # traj = torch.from_numpy(file['trajs'][index])
            image = file['images'][index]

        # Applying transformation
        if self.transform is not None:
            image = self.transform(image)

        inp = image[0:self.inp_len:self.slice]
        trg = image[self.inp_len:self.inp_len + self.trg_len:self.slice]

        return inp, trg
