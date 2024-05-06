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


class Predictor(nn.Module):
    def __init__(self,
                 n_inputs: int,
                 n_states: int,
                 n_hidden: int,
                 n_layers: int,
                 n_outputs: int = 1) -> None:
        super().__init__()
        self.n_inputs = n_inputs
        self.n_states = n_states
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_outputs = n_outputs

        self.h = self.network(n_inputs, n_states, n_hidden, n_layers, n_outputs)

    @staticmethod
    def network(n_inputs: int,
                n_states: int,
                n_hidden: int,
                n_layers: int,
                output_dim: int = 1
                ) -> nn.Sequential:
        def block(in_feat, out_feat, normalize=True, dropout=0.0):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.LayerNorm(out_feat))
            if dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU(inplace=True))
            return layers

        net = nn.Sequential(
            # input layer
            *block(n_states + n_inputs, n_hidden),

            # hidden layers
            *[module for _ in range(n_layers - 1) for module in block(n_hidden, n_hidden)],

            # output layer
            nn.Linear(n_hidden, output_dim),
        )

        return net

    # Assume inputs are (x_1, x_2, ... , x_n, u_1, u_2, ... , u_m, du_1, du_2, ... , du_m)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.h(x)
