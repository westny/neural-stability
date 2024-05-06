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
from torch import nn
import lightning.pytorch as pl
import torch.distributions as dist

mse_loss = nn.MSELoss(reduction='none')


class LitModel(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 config: dict) -> None:
        super().__init__()
        self.model = model
        self.lr = config["lr"]
        self.beta = config["beta"]
        self.decay = config["decay"]
        self.dataset = config["data"]
        self.sample_time = config["sample_time"]
        self.max_epochs = config["epochs"]
        self.save_hyperparameters(ignore=['model'])

    def forward(self, *args, **kwargs) -> None:
        pass

    @staticmethod
    def data_info(target: torch.Tensor) -> tuple[int, int]:
        # return target.shape[0:2]
        size = target.size()
        return size[0], size[1]

    def training_step(self,
                      data: tuple[torch.Tensor, torch.Tensor],
                      batch_idx: int
                      ) -> torch.Tensor:
        inputs, target = data

        int_steps, batch_size = self.data_info(target)
        pred, _, q = self.model(inputs, int_steps, self.sample_time, self.training)

        kl_div = dist.kl_divergence(q, dist.Normal(0, 1)).sum(-1).mean()
        recon_loss = mse_loss(pred, target).sum([-2, -1]).mean()

        loss = recon_loss + self.beta * kl_div

        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=batch_size, prog_bar=True)
        return loss

    def validation_step(self,
                        data: tuple[torch.Tensor, torch.Tensor],
                        batch_idx: int
                        ) -> torch.Tensor:
        inputs, target = data
        int_steps, batch_size = self.data_info(target)
        pred, *_ = self.model(inputs, int_steps, self.sample_time, self.training)
        loss = mse_loss(pred, target).mean()
        self.log("val_loss", loss, on_epoch=True, batch_size=batch_size, prog_bar=True)
        return loss

    def test_step(self,
                  data: tuple[torch.Tensor, torch.Tensor],
                  batch_idx: int
                  ) -> torch.Tensor:
        inputs, target = data
        int_steps, batch_size = self.data_info(target)
        pred, *_ = self.model(inputs, int_steps, self.sample_time, self.training)
        loss = mse_loss(pred, target).mean()
        self.log("test_loss", loss, on_epoch=True, batch_size=batch_size, prog_bar=True)
        return loss

    def configure_optimizers(self
                             ) -> tuple[list[torch.optim.Optimizer],
                                        list[torch.optim.lr_scheduler.LRScheduler]]:
        optimizer = torch.optim.NAdam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.decay)
        return [optimizer], [scheduler]
