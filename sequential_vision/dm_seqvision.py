import os
import torch

import lightning.pytorch as pl
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader


class PermuteTransform:
    def __init__(self):
        self.perm_idx = torch.randperm(28 * 28)

    def __call__(self, tensor):
        # Assuming tensor is of size (1, 28, 28) for grayscale MNIST images
        return tensor.view(-1)[self.perm_idx].view(1, 28, 28)


class LitDataModule(pl.LightningDataModule):
    def __init__(self, args, config: dict):
        super().__init__()
        root = config["root"]
        self.dataset = config["name"]

        self.data_aug = config["augment"]
        self.batch_size = config["batch_size"]
        self.grey_scale = config["grey_scale"]
        self.pixel_wise = config["pixel_wise"]

        self.n_workers = args.n_workers
        self.pin_memory = args.pin_memory
        self.persistent = args.persistent_workers

        assert self.dataset in ['mnist', 'pmnist', 'cifar10'], "Dataset not supported"

        if self.dataset == 'mnist':
            transform_list = [
                transforms.ToTensor(),
                transforms.Normalize(mean=0.1307, std=0.3081)
            ]
            if self.pixel_wise:
                transform_list.append(transforms.Lambda(lambda x: x.view(28 * 28)))
            else:
                transform_list.append(transforms.Lambda(lambda x: x.view(28, 28)))

            transform = transforms.Compose(transform_list)

            self.train = datasets.MNIST(root, train=True, download=True, transform=transform)
            self.test = datasets.MNIST(root, train=False, download=True, transform=transform)

        elif self.dataset == 'pmnist':
            transform_list = [
                transforms.ToTensor(),
                PermuteTransform(),
                transforms.Normalize(mean=0.1307, std=0.3081)
            ]
            if self.pixel_wise:
                transform_list.append(transforms.Lambda(lambda x: x.view(28 * 28)))
            else:
                transform_list.append(transforms.Lambda(lambda x: x.view(28, 28)))

            transform = transforms.Compose(transform_list)

            self.train = datasets.MNIST(root, train=True, download=True, transform=transform)
            self.test = datasets.MNIST(root, train=False, download=True, transform=transform)

        elif self.dataset == 'cifar10':
            if self.grey_scale:
                transform_list = [
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=122.6 / 255.0, std=61.0 / 255)
                ]
                if self.pixel_wise:
                    transform_list.append(transforms.Lambda(lambda x: x.view(32 * 32)))
                else:
                    transform_list.append(transforms.Lambda(lambda x: x.view(32, 32)))
            else:
                transform_list = [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2616))
                ]
                if self.pixel_wise:
                    transform_list.append(transforms.Lambda(lambda x: x.view(3, 32 * 32).t()))
                else:
                    transform_list.append(transforms.Lambda(lambda x: x.permute(1, 0, 2).flatten(1)))

            transform = transforms.Compose(transform_list)
            self.test = datasets.CIFAR10(root, train=False, download=True, transform=transform)

            if self.data_aug:
                transform_list.insert(0, transforms.RandomHorizontalFlip(p=0.5))
                transform_list.insert(0, transforms.RandomCrop(32, padding=4))
                transform = transforms.Compose(transform_list)

            self.train = datasets.CIFAR10(root, train=True, download=True, transform=transform)

        else:
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(self.train,
                          shuffle=True,
                          batch_size=self.batch_size,
                          num_workers=self.n_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent)

    def val_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.n_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent)

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.n_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent)
