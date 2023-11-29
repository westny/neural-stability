import os
import h5py
import torch
import lightning.pytorch as pl
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader


class LitDataModule(pl.LightningDataModule):
    def __init__(self, args, config: dict):
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

        self.data_path = f"{self.root}/{self.dataset}/"
        # check if /data/args.dataset exists, else raise error
        if not os.path.exists(self.data_path):
            raise ValueError(f"Could not find data. Please run tf_converter_hdf5.py to download the data.")

        self.train = CustomHDF5Dataset(self.data_path, "train",
                                       self.inp_len, self.trg_len, self.sample_time)
        self.val = CustomHDF5Dataset(self.data_path, "val",
                                     self.inp_len, self.trg_len, self.sample_time)
        self.test = CustomHDF5Dataset(self.data_path, "test",
                                      self.inp_len, self.test_trg_len, self.sample_time)

    def train_dataloader(self):
        return DataLoader(self.train,
                          shuffle=True,
                          batch_size=self.batch_size,
                          num_workers=self.n_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers,
                          prefetch_factor=4,
                          )

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          num_workers=self.n_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers
                          )

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.n_workers)


class SequentialToTensor:
    def __call__(self, pic):
        """
        Convert a numpy.ndarray with shape (S, H, W, C) to torch.FloatTensor with shape (S, C, H, W).
        Pixel values are scaled from [0, 255] to [0.0, 1.0].
        """
        tensor = torch.from_numpy(pic.transpose((0, 3, 1, 2)))

        return tensor.float().div(255)


class CustomHDF5Dataset(Dataset):
    def __init__(self,
                 hdf5_file_path: str = "data/twoXpendulum/",
                 train_test: str = "train",
                 inp_len: int = 10,
                 trg_len: int = 60,
                 sample_time: float = 0.05):
        """
        Args:
            hdf5_file_path (string): Path to the HDF5 file.
            train_test (string): train, val or test.
            inp_len (int): Number of input steps.
            trg_len (int): Number of target steps.
            sample_time (float): Sampling time of the data.
        """
        self.file_path = hdf5_file_path + train_test + ".hdf5"
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

    def __len__(self):
        return self.length

    def __getitem__(self, index):
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
