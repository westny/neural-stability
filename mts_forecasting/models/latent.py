import torch.nn as nn


class Latent(nn.Module):
    def __init__(self,
                 n_inputs: int,
                 n_states: int,
                 n_hidden: int,
                 n_layers: int):
        super().__init__()

        self.g = self.network(n_inputs, n_states, n_hidden, n_layers)

    @staticmethod
    def network(n_inputs, n_states, n_hidden, n_layers):
        def block(in_feat, out_feat, normalize=True, dropout=0.0):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.LayerNorm(out_feat))
            if dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Tanh())
            return layers

        net = nn.Sequential(
            *block(n_inputs, n_hidden),

            # hidden layers
            *[module for _ in range(n_layers - 1) for module in block(n_hidden, n_hidden)],

            # output layer
            nn.Linear(n_hidden, 2 * n_states),
        )

        return net

    def forward(self, x):
        return self.g(x)
