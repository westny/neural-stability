import torch
from torch.utils.data import Dataset


class MTSDataset(Dataset):
    def __init__(self, inp_data, trg_data):
        self.input = inp_data
        self.trg_data = trg_data
        self.inp_idx = torch.arange(len(inp_data))

    @staticmethod
    def collate_fn(batch):
        x = torch.stack([b[0] for b in batch], dim=1)
        y = torch.stack([b[1] for b in batch], dim=1)
        return x, y

    def __len__(self):
        return len(self.inp_idx)

    def __getitem__(self, idx):
        x = self.input[idx]
        y = self.trg_data[idx]
        return x, y
