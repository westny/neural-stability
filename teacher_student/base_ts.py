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
from lightning.pytorch import LightningModule

huber_loss = nn.HuberLoss()
mse_loss = nn.MSELoss()


class LitModel(LightningModule):
    def __init__(self, model: nn.Module, config: dict) -> None:
        super().__init__()
        self.model = model
        self.max_epochs = config['epochs']
        self.learning_rate = config['lr']

        self.save_hyperparameters(ignore=['model'])

        self.val_steps: list[float] = []
        self.val_progress: list[float] = []
        self.pole_progress: list[torch.Tensor] = []

    def forward(self, *args, **kwargs) -> None:
        pass

    @staticmethod
    def post_process(data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        target, inp, time = data
        y0 = target[0, :]
        seq_len = target.shape[0]
        batch_size = target.shape[1]
        return y0, target, inp, time, seq_len, batch_size

    def training_step(self,
                      data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                      batch_idx: int
                      ) -> torch.Tensor:
        y0, target, inp, t, seq_len, batch_size = self.post_process(data)

        if self.model.f.controlled:
            pred = []
            ti = t[0:2]
            for si in range(seq_len):
                ui = inp[si]
                y0 = self.model(ti, y0, ui)
                y0 = y0[-1]
                pred.append(y0)
            preds = torch.stack(pred, dim=0)
        else:
            preds = self.model(t, y0).squeeze()

        loss = mse_loss(preds, target)
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size, prog_bar=True)
        return loss

    def validation_step(self,
                        data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                        batch_idx: int
                        ) -> torch.Tensor:
        y0, target, inp, t, seq_len, batch_size = self.post_process(data)

        if self.model.f.controlled:
            pred = []
            ti = t[0:2]
            for si in range(seq_len):
                ui = inp[si]
                y0 = self.model(ti, y0, ui)
                y0 = y0[-1]
                pred.append(y0)
            preds = torch.stack(pred, dim=0)
        else:
            preds = self.model(t, y0).squeeze()
        loss = mse_loss(preds, target)
        self.val_steps.append(loss.item())

        # self.log("val_loss", loss, on_epoch=True, batch_size=batch_size, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        loss = sum(self.val_steps) / len(self.val_steps)
        self.val_steps.clear()
        self.val_progress.append(loss)
        self.estimate_zero_poles()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def estimate_zero_poles(self) -> None:
        self.model.eval()
        y0 = torch.zeros((1, self.model.n_states), device=self.device)
        u0 = torch.zeros((1, self.model.n_inputs), device=self.device)
        jac = self.model.state_jacobian(y0, u0)
        eigs = torch.linalg.eigvals(jac).detach()
        poles = torch.stack((eigs.real, eigs.imag), dim=0)
        self.pole_progress.append(poles)

    def estimate_poles(self,
                       y0: torch.Tensor,
                       t: torch.Tensor,
                       inp: torch.Tensor
                       ) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            y0 = y0.to(self.device)
            t = t.to(self.device)
            inp = inp.to(self.device)

            if self.model.f.controlled:
                pred = []
                ti = t[0: 2]
                for si in range(len(t)):
                    ui = inp[si:si + 1]

                    y0 = self.model(ti, y0, ui)
                    y0 = y0[-1]
                    pred.append(y0)
                test_y = torch.cat(pred, dim=0)
            else:
                test_y = self.model(t, y0).squeeze()
        # Calculate eigenvalues based on the linearized system
        eigen_real = []
        eigen_imag = []
        for si in range(len(t)):
            # for yi, ui in zip(test_y, inp):
            yi, ui = test_y[si], inp[si]
            jac = self.model.state_jacobian(yi[None], ui[None])
            eig_i = torch.linalg.eigvals(jac)
            eigen_real.append(torch.real(eig_i).squeeze().detach())
            eigen_imag.append(torch.imag(eig_i).squeeze().detach())

        eigen_reals = torch.stack(eigen_real).cpu()
        eigen_imags = torch.stack(eigen_imag).cpu()

        # eigenvalues = {'re': eigen_real, 'im': eigen_imag}
        poles = torch.stack((eigen_reals, eigen_imags), dim=0)
        return poles

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.NAdam(self.parameters(), betas=(0.9, 0.999), lr=self.learning_rate)
        # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.1,
        #                                               total_iters=self.max_epochs)
        # return [optimizer], [scheduler]
        return optimizer
