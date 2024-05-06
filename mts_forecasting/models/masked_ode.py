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
from typing import Optional
from models.neuralode import NeuralODE


class MaskedODE(NeuralODE):
    def __init__(self,
                 n_states: int,
                 n_inputs: int,
                 n_hidden: int,
                 n_layers: int,
                 solver: str,
                 h: float,
                 stability_init: bool,
                 activation: str = "swish") -> None:
        super().__init__(n_states,
                         n_inputs,
                         n_hidden,
                         n_layers,
                         solver,
                         h,
                         stability_init=stability_init,
                         activation=activation)

        self.n_states = n_states
        self.n_inputs = n_inputs

    def model_update(self,
                     t: torch.Tensor,
                     X: tuple[torch.Tensor, torch.Tensor]
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        x, u = X
        dx = self.f((x, u))
        du = u * 0
        return dx, du

    def forward(self,
                t: torch.Tensor,
                x: torch.Tensor,
                u: Optional[torch.Tensor] = None) -> torch.Tensor:
        return super().forward(t, x, u=u)
