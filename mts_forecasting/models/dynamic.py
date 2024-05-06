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
from typing import Any, Optional

import torch
from torch import nn
import torch.distributions as dist

# NN block
from mts_forecasting.models.masked_ode import MaskedODE as NeuralODE
from mts_forecasting.models.latent import Latent
from mts_forecasting.models.predictor import Predictor


class MultiRegressor(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.n_inputs = config["n_inputs"]
        self.n_states = config["n_latent"]
        self.n_outputs = config["n_outputs"]

        self.n_hidden = config["n_hidden"]
        self.n_layers = config["n_layers"]

        self.g = Latent(self.n_inputs, self.n_states, self.n_hidden, self.n_layers)
        self.h = Predictor(0, self.n_states, self.n_hidden, self.n_layers, self.n_outputs)
        self.f = NeuralODE(n_states=self.n_states,
                           n_inputs=self.n_inputs,
                           n_hidden=self.n_hidden,
                           n_layers=self.n_layers,
                           solver=config["ode_solver"],
                           h=config["step_size"],
                           stability_init=config["stability_init"],
                           activation=config["activation"])

    def sample(self,
               u: torch.Tensor,
               variational: bool = True,
               training: bool = True
               ) -> tuple[torch.Tensor, Optional[dist.Distribution]]:
        if variational:
            if training:
                out = self.g(u[:, :self.n_inputs])
                qz0_mean, qz0_logvar = out[:, :self.n_states], out[:, self.n_states:]
                q = dist.Normal(qz0_mean, torch.exp(qz0_logvar / 2.))
                z0 = q.rsample()
            else:
                batch_size = u.shape[0]
                q = dist.Normal(torch.zeros(batch_size, self.n_states, device=u.device),
                                torch.ones(batch_size, self.n_states, device=u.device))
                z0 = q.rsample()
        else:
            z0 = self.g(u[:, :self.n_inputs])[:, :self.n_states]
            q = None
        return z0, q

    def forward(self,
                u: torch.Tensor,
                T: int,
                dt: float = 0.05,
                training: bool = True
                ) -> tuple[torch.Tensor, torch.Tensor, Optional[dist.Distribution]]:
        """
        N = batch size

        Parameters
        ----------
        u: torch.Tensor [T, N, n_inputs * 2]
        T: int
        dt: float
        training: bool

        Returns
        -------
        y_pred: torch.Tensor [T, N, n_outputs]
        states: torch.Tensor [T, N, n_states]
        q: dist.Distribution

        """

        zt, q = self.sample(u[0], training=training)

        # Store predictions
        states = []

        # Init time
        t0 = 0.
        t_vec = torch.tensor([t0, t0 + dt], device=u.device)

        for t in range(0, T):
            zt = self.f(t_vec, zt, u[t])
            zt = zt[-1, :, :self.n_states]
            states.append(zt)

        stacked_states = torch.stack(states, dim=0)

        y_pred = self.h(stacked_states)

        return y_pred, stacked_states, q
