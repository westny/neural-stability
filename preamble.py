import os
import time
import json
import torch
import warnings
import importlib
from pathlib import Path

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

torch.set_float32_matmul_precision('medium')
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings("ignore", ".*Checkpoint directory*")


def load_config(config):
    # check if file contains ".json" extension
    if not config.endswith(".json"):
        config += ".json"

    # check if file exists in any of the config subdirectories
    config_path = Path("configs")

    # get all subdirectories
    subdirs = [d for d in config_path.iterdir() if d.is_dir()]

    # get all files in subdirectories
    files = [f for d in subdirs for f in d.iterdir() if f.is_file()]

    # check if config is any of the files
    if not any([config in f.name for f in files]):
        raise FileNotFoundError(f"Config file {config} not found.")
    else:
        config = [f for f in files if config in f.name][0]

    with open(config, 'r') as openfile:
        conf = json.load(openfile)
    return conf


def import_module(module_name):
    return importlib.import_module(module_name)


def import_from_module(module_name, class_name):
    module = import_module(module_name)
    return getattr(module, class_name)
