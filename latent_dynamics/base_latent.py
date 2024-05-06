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

import torch.nn as nn
import lightning.pytorch as pl
from latent_dynamics.utils import *

mse_loss = nn.MSELoss(reduction='none')


class LitModel(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 config: dict
                 ) -> None:
        super().__init__()
        self.model = model
        self.epochs = config["epochs"]
        self.learning_rate = config["lr"]
        self.sample_time = config["sample_time"]
        self.save_hyperparameters(ignore=['model'])

    def forward(self, *args, **kwargs) -> None:
        pass

    def post_process(self,
                     batch: tuple[torch.Tensor, torch.Tensor]
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inp, gt = batch
        trg_len = gt.shape[1]
        t = torch.linspace(0, trg_len * self.sample_time, trg_len + 1, device=self.device)
        return inp, gt, t

    def training_step(self,
                      data: tuple[torch.Tensor, torch.Tensor],
                      batch_idx: int
                      ) -> torch.Tensor:
        inp, gt, t = self.post_process(data)
        pred = self.model(t, inp)
        loss = mse_loss(pred, gt).sum(1).mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=inp.shape[0], prog_bar=True)
        return loss

    def validation_step(self,
                        data: tuple[torch.Tensor, torch.Tensor],
                        batch_idx: int
                        ) -> torch.Tensor:
        inp, gt, t = self.post_process(data)
        pred = self.model(t, inp)

        mse = mse_loss(pred, gt)
        loss = mse.sum(1).mean()
        norm_loss = (loss / pred.shape[1]) / gt.square().mean()
        accuracy = pixel_accuracy(pred, gt)

        log_dict = {"val_acc": accuracy, "norm_loss": norm_loss}

        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self,
                  data: tuple[torch.Tensor, torch.Tensor],
                  batch_idx: int
                  ) -> torch.Tensor:
        inp, gt, t = self.post_process(data)
        pred = self.model(t, inp)

        loss = mse_loss(pred, gt).mean() / gt.square().mean()
        accuracy = pixel_accuracy(pred, gt)

        log_dict = {"norm_mse": loss, "test_acc": accuracy}

        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self
                             ) -> tuple[list[torch.optim.Optimizer],
                                        list[torch.optim.lr_scheduler.LRScheduler]]:
        optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.999), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1,
                                                      total_iters=self.epochs)
        return [optimizer], [scheduler]
