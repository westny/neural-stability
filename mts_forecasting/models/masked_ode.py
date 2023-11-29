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
                 activation: str = "swish"):
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

    def model_update(self, t, X):
        x, u = X
        dx = self.f((x, u))
        du = u * 0
        return dx, du

    def forward(self, t, x, u=None):
        return super().forward(t, x, u=u)
