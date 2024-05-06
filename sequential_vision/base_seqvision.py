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
import torch.nn as nn
import lightning.pytorch as pl
from torch.nn import functional as F


class LitModel(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 config: dict) -> None:
        super().__init__()
        self.model = model
        self.epochs = config["epochs"]
        self.learning_rate = config["lr"]
        self.sample_time = config["sample_time"]

        self.save_hyperparameters(ignore=['model'])

    def forward(self, *args, **kwargs) -> None:
        pass

    def training_step(self,
                      data: tuple[torch.Tensor, torch.Tensor],
                      batch_idx: int
                      ) -> torch.Tensor:
        inputs, target = data
        if inputs.dim() == 2:
            inputs.unsqueeze_(-1)
        batch_size, seq_len, feat_dim = inputs.shape

        ti = torch.tensor([0, self.sample_time], device=self.device)
        x0 = torch.zeros((batch_size, self.model.n_states), device=self.device)

        inputs = self.model.encoder(inputs)

        loss = torch.tensor(0., device=self.device)
        scales = torch.linspace(0.1, 1, seq_len) ** 4

        for si in range(seq_len):
            ui = inputs[:, si]
            x0 = self.model.ode(ti, x0, ui)
            x0 = x0[-1]
            y_hat = self.model.decoder(x0)
            loss += F.cross_entropy(y_hat, target) * scales[si]

        accuracy = (y_hat.argmax(1) == target).float().mean()
        log_dict = {'train_loss': loss,
                    'train_accuracy': accuracy}
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self,
                        data: tuple[torch.Tensor, torch.Tensor],
                        batch_idx: int
                        ) -> torch.Tensor:
        inputs, target = data
        if inputs.dim() == 2:
            inputs.unsqueeze_(-1)
        batch_size, seq_len, feat_dim = inputs.shape
        ti = torch.tensor([0, self.sample_time], device=self.device)
        x0 = torch.zeros((batch_size, self.model.n_states), device=self.device)

        inputs = self.model.encoder(inputs)

        for si in range(seq_len):
            ui = inputs[:, si]
            x0 = self.model.ode(ti, x0, ui)
            x0 = x0[-1]
        y_hat = self.model.decoder(x0)

        loss = F.cross_entropy(y_hat, target)

        accuracy = (y_hat.argmax(1) == target).float().mean()

        metrics = {'test_loss': loss,
                   'test_accuracy': accuracy}

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self
                             ) -> tuple[list[torch.optim.Optimizer],
                                        list[torch.optim.lr_scheduler.LRScheduler]]:
        optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.999), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.1,
                                                      total_iters=self.epochs)
        return [optimizer], [scheduler]
