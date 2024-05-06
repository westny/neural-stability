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
import sys
import h5py
import time
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from multiprocessing import Process, Manager
from latent_dynamics.data_utils import *

parser = ArgumentParser(description='Dataset converter arguments')
parser.add_argument('--dataset', type=str, default="mujoco",
                    help='dataset to convert. Options: spring, pendulum, twoXpendulum, two_bodies, mujoco.'
                         ' Default: mujoco')
args = parser.parse_args()

cwd = os.getcwd()
cf = os.path.basename(cwd)
this_folder = "latent_dynamics"

if cf == this_folder:
    root = os.path.join(os.getcwd(), '../data')
elif os.path.isdir(os.path.join(cwd, this_folder)):
    root = os.path.join(os.getcwd(), './data')
else:
    raise ValueError(f"Unknown project root. Please run script from either {this_folder} or the main project folder")

# check if /data/args.dataset exists, create empty folder if not
if not os.path.exists(f"{root}/{args.dataset}"):
    os.makedirs(f"{root}/{args.dataset}")
else:
    print(f"Folder data/{args.dataset} already exists.")
    inp = input("Would you like to overwrite it? (y/n) \n").lower()
    if inp == "y":
        print("Overwriting folder...")
        # Clear folder
        for filename in os.listdir(f"{root}/{args.dataset}"):
            file_path = os.path.join(f"{root}/{args.dataset}", filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        print("Exiting...")
        exit()

# Get url, path, and config for dataset
url, path, config = switch(args.dataset, root)
train_path = path + "_train.tfrecord"
test_path = path + "_test.tfrecord"


# Check if data files exist, download if not
if not os.path.exists(train_path):
    inp = input("Could not find data. Would you like to download it? (y/n) \n").lower()
    if inp == "y":
        print("Downloading training data...")
        download_file(url + "train.tfrecord", train_path)
        print("Downloading test data...")
        download_file(url + "test.tfrecord", test_path)
    else:
        print("Exiting...")
        sys.exit()
    print("Finished downloading data.")

collate_fn = lambda x: collate_fn_conv(x, config)

val_split = 0.2

train_samples = 50000
test_samples = 20000
val_samples = int(train_samples * val_split)
train_samples -= val_samples

# Create generators for each split using known indices
train_gen = tfrecord_generator(train_path, 0, train_samples, collate_fn)
val_gen = tfrecord_generator(train_path, train_samples, train_samples + val_samples, collate_fn)
test_gen = tfrecord_generator(test_path, 0, test_samples, collate_fn)

print(f"\nPreparing to convert {args.dataset} data set to .hdf5 format... \n")


def add_to_hdf5(hdf5_dataset, data, i):
    hdf5_dataset[i] = data


def create_hdf5_dataset(file, name, shape, dtype):
    hdf5_dataset = file.create_dataset(name, shape=shape, dtype=dtype)
    return hdf5_dataset


def loop_over_samples(n_samples, img_dataset, traj_dataset, iterator, progress_queue):
    for i in range(n_samples):
        trajectory, image = next(iterator)
        add_to_hdf5(traj_dataset, trajectory, i)
        add_to_hdf5(img_dataset, image, i)
        progress_queue.put(1)


def create_dataset(samples, name, iterator, progress_queue):
    with h5py.File(f"{root}/{args.dataset}/{name}.hdf5", "w") as f:
        traj_shape = (samples,) + config["x"][1]
        img_shape = (samples,) + config["image"][1]

        img_ds = create_hdf5_dataset(f, "images", img_shape, np.dtype('uint8'))
        traj_ds = create_hdf5_dataset(f, "trajs", traj_shape, np.float32)
        loop_over_samples(samples, img_ds, traj_ds, iterator, progress_queue)


def listener(total, q, description, position):
    pbar = tqdm(total=total, desc=description, position=position,
                dynamic_ncols=True, file=sys.stdout)
    for _ in iter(q.get, None):
        pbar.update()
        sys.stdout.flush()
    pbar.close()


if __name__ == "__main__":
    manager = Manager()

    train_queue = manager.Queue()
    val_queue = manager.Queue()
    test_queue = manager.Queue()
    queues = [train_queue, val_queue, test_queue]

    # listeners
    train_listener = Process(target=listener, args=(train_samples, train_queue, "Training", 0))
    val_listener = Process(target=listener, args=(val_samples, val_queue, "Validation", 1))
    test_listener = Process(target=listener, args=(test_samples, test_queue, "Testing", 2))
    listeners = [train_listener, val_listener, test_listener]

    train_process = Process(target=create_dataset, args=(train_samples, "train", train_gen, train_queue))
    val_process = Process(target=create_dataset, args=(val_samples, "val", val_gen, val_queue))
    test_process = Process(target=create_dataset, args=(test_samples, "test", test_gen, test_queue))
    processes = [train_process, val_process, test_process]

    try:
        for listener in listeners:
            listener.start()

        # Ensure the listeners have started
        time.sleep(1)

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        for queue in queues:
            queue.put(None)

        for listener in listeners:
            listener.join()
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Terminating processes...")
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()

        for queue in queues:
            queue.put(None)

        for listener in listeners:
            if listener.is_alive():
                listener.terminate()
                listener.join()
        sys.exit()
    else:
        print("Finished converting data set to .hdf5 format.")
