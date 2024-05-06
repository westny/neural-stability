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


import time
import pathlib
import warnings
import torch

from torch.multiprocessing import set_sharing_strategy
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import Logger, WandbLogger
from lightning.pytorch.strategies import Strategy, DDPStrategy

from arguments import args
from preamble import load_config, import_from_module

torch.set_float32_matmul_precision('medium')
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings("ignore", ".*Checkpoint directory*")

set_sharing_strategy('file_system')

# Load configuration and import modules
config = load_config(args.config)
TorchModel = import_from_module(config["model"]["module"], config["model"]["class"])
LitDataModule = import_from_module(config["datamodule"]["module"], config["datamodule"]["class"])
LitModel = import_from_module(config["litmodule"]["module"], config["litmodule"]["class"])


def main(save_name: str):
    ds = config["dataset"]
    ckpt_path = pathlib.Path(f"saved_models/{ds}/{save_name}.ckpt")

    # Check if checkpoint exists and the overwrite flag is not set
    if ckpt_path.exists() and not args.overwrite:
        ckpt = str(ckpt_path)
    else:
        ckpt = None

    # Setup callbacks list for training
    callback_list: list[Callback] = []
    if args.store_model:
        ckpt_cb = ModelCheckpoint(
            dirpath=str(ckpt_path.parent),  # Using parent directory of the checkpoint
            filename=save_name + "_{epoch:02d}",
        )

        ckpt_cb_best = ModelCheckpoint(
            dirpath=str(ckpt_path.parent),
            filename=save_name,
            monitor="val_loss",
            mode="min"
        )

        callback_list += [ckpt_cb, ckpt_cb_best]

    # Determine the number of devices, strategy and accelerator
    strategy: str | Strategy
    if torch.cuda.is_available() and args.use_cuda:
        devices = -1 if torch.cuda.device_count() > 1 else 1
        strategy = DDPStrategy(find_unused_parameters=True,
                               gradient_as_bucket_view=True) if devices == -1 else 'auto'
        accelerator = "auto"
    else:
        devices, strategy, accelerator = 1, 'auto', "cpu"

    # Setup logger
    logger: bool | Logger
    if args.dry_run:
        logger = False
        args.small_ds = True
    elif not args.use_logger:
        logger = False
    else:
        run_name = f"{save_name}_{time.strftime('%d-%m_%H:%M:%S')}"
        task = config["task"]
        logger = WandbLogger(project=f"neural-stability-{task}", name=run_name)

    clip_val = config["training"]["clip"] if config["training"]["clip"] else None

    # Setup model, datamodule and trainer
    model = TorchModel(config["model"])
    datamodule = LitDataModule(args, config["datamodule"])
    lit_model = LitModel(model, config["training"])

    # Trainer configuration
    trainer = Trainer(max_epochs=config["training"]["epochs"],
                      logger=logger,
                      devices=devices,
                      strategy=strategy,
                      accelerator=accelerator,
                      callbacks=callback_list,
                      gradient_clip_val=clip_val,
                      fast_dev_run=args.dry_run,
                      enable_checkpointing=args.store_model)

    # Model fitting
    trainer.fit(lit_model, datamodule=datamodule, ckpt_path=ckpt)


if __name__ == "__main__":
    seed_everything(args.main_seed, workers=True)

    if args.scnd_seed is not None:
        config["datamodule"]["data_seed"] = args.scnd_seed

    if args.dry_run:
        config["datamodule"]["batch_size"] = 2

    config["model"]["stability_init"] = args.stability_init

    if args.add_name:
        add_name = f"_{args.add_name}"
    else:
        add_name = ""

    stability = "SII" if config["model"]["stability_init"] else "He"

    model_name = f'{config["task"]}_{config["dataset"]}_{stability}{add_name}'

    print(f'Preparing to train model: {model_name}')

    main(model_name)
