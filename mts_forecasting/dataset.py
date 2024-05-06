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

import torch
from torch.utils.data import Dataset


class MTSDataset(Dataset):
    def __init__(self, inp_data, trg_data) -> None:
        self.input = inp_data
        self.trg_data = trg_data
        self.inp_idx = torch.arange(len(inp_data))

    @staticmethod
    def collate_fn(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.stack([b[0] for b in batch], dim=1)
        y = torch.stack([b[1] for b in batch], dim=1)
        return x, y

    def __len__(self) -> int:
        return len(self.inp_idx)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input[idx]
        y = self.trg_data[idx]
        return x, y
