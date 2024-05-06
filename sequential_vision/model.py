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
from models.neuralode import NeuralODE


class PixelLevelClassifier(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.n_inputs = config["n_inputs"]
        self.n_states = config["n_latent"]
        n_hidden = config["n_hidden"]
        n_layers = config["n_layers"]
        n_encodings = config["n_encodings"]
        n_outputs = config["n_outputs"]

        self.encoder = nn.Linear(self.n_inputs, n_encodings)
        self.decoder = nn.Sequential(
            nn.Linear(self.n_states, self.n_states // 2),
            nn.ReLU(),
            nn.Linear(self.n_states // 2, n_outputs)
        )

        self.ode = NeuralODE(n_states=self.n_states,
                             n_inputs=n_encodings,
                             n_hidden=n_hidden,
                             n_layers=n_layers,
                             solver=config["ode_solver"],
                             h=config["step_size"],
                             stability_init=config["stability_init"],
                             activation=config["activation"])

    def forward(self, *args):
        pass
