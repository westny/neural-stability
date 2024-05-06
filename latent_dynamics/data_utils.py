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
from typing import Any, Callable

import numpy as np
import torch
import torchvision
import requests
from tfrecord.torch.dataset import TFRecordDataset
from latent_dynamics.configs import *

main_url = "https://storage.googleapis.com/dm-hamiltonian-dynamics-suite/"


def download_file(url: str, filename: str) -> None:
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size_in_bytes = int(r.headers.get('content-length', 0))
        total_size_in_gb = total_size_in_bytes / (1024 * 1024 * 1024)  # Convert to GB
        block_size = 10 * 1024 * 1024  # 100 MB
        progress = 0

        with open(filename, 'wb') as file:
            for data in r.iter_content(block_size):
                file.write(data)
                file.flush()
                os.fsync(file.fileno())
                progress += len(data)
                downloaded_in_gb = progress / (1024 * 1024 * 1024)  # Convert to GB
                done = int(50 * progress / total_size_in_bytes)
                print(f"\r[{'=' * done}{' ' * (50 - done)}] {downloaded_in_gb:.2f}/{total_size_in_gb:.2f} GB", end='')
        print()


def switch(dataset: str,
           root: str = "data") -> tuple[str, str, dict]:
    if dataset == "spring":
        return (main_url + "toy_physics/mass_spring/",
                f"{root}/toy_physics_mass_spring",
                config_mass_spring)
    elif dataset == "pendulum":
        return (main_url + "toy_physics/pendulum/",
                f"{root}/toy_physics_pendulum",
                config_pendulum)
    elif dataset == "twoXpendulum":
        return (main_url + "toy_physics/double_pendulum/",
                f"{root}/toy_physics_double_pendulum",
                config_double_pendulum)
    elif dataset == "two_bodies":
        return (main_url + "toy_physics/two_bodies/",
                f"{root}/toy_physics_two_body",
                config_two_bodies)
    elif dataset == "mujoco":
        return (main_url + "mujoco_room/circle/",
                f"{root}/mujoco_room_circle",
                config_mujoco_room)
    elif dataset == "molecules":
        return (main_url + "molecular_dynamics/lj_16/",
                f"{root}/molecular_dynamics_lj_16",
                config_molecules)
    else:
        raise NotImplementedError


def tfrecord_generator(tfrecord_path: str,
                       start_idx: int,
                       end_idx: int,
                       transform_fn: Callable) -> Any:
    dataset = TFRecordDataset(tfrecord_path, None)
    for idx, data in enumerate(dataset):
        if idx < start_idx:
            continue
        elif idx >= end_idx:
            break
        else:
            data = transform_fn(data)
            yield data


def collate_fn_img(batch: dict, conf: dict) -> np.ndarray:
    dtype, shape = conf["image"]
    value = batch["image"]
    uint8_tensor = torch.tensor([i for i in value], dtype=torch.uint8)
    img = torchvision.io.decode_png(uint8_tensor)
    data = img.view(3, shape[0], 32, 32).permute(1, 2, 3, 0)
    data = data.numpy()
    return data


def collate_fn_state(batch: dict, conf: dict) -> torch.Tensor:
    dtype, shape = conf["x"]
    value = batch["x"]
    data = torch.tensor(value).view(torch.float64).float()
    data = data.reshape(shape)
    return data


def collate_fn_conv(batch: dict, conf: dict) -> tuple[torch.Tensor, np.ndarray]:
    trajectory = collate_fn_state(batch, conf)
    images = collate_fn_img(batch, conf)
    return trajectory, images
