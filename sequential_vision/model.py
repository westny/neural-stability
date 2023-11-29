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
