import torch
import torch.nn as nn

from torch import func
from torchdiffeq import odeint
from teacher_student.torch_interp1d import interp1d
from stb_init import init_strategy


class ReferenceSystem(nn.Module):
    n_inputs: int = 1
    solver: str = "dopri5"

    def __init__(self):
        super().__init__()
        if not self.n_inputs > 0:
            self._rgi = lambda *args: 0.
        else:
            self._rgi = lambda *args: torch.zeros((1, self.n_inputs))

    @property
    def rgi(self):
        return self._rgi

    def reset_rgi(self):
        self._rgi = lambda *args: torch.zeros((1, self.n_inputs))

    def forward(self, t, y):
        return y

    def generate_input(self, t, u="square"):
        from scipy.signal import sawtooth, square, chirp
        import numpy as np

        if u == "square":
            inp = torch.tensor(square(t)).float()
        elif u == "pwm_sine":
            sig = torch.sin(torch.pi * t / 8)
            pwm = square(torch.pi * t, duty=(sig + 1) / 2)
            noise = np.random.normal(0, 0.1, size=t.shape)
            inp = torch.tensor(pwm + noise).float()
        elif u == "pwm_cosine":
            sig = torch.cos(torch.pi * t / 8)
            pwm = square(torch.pi * t, duty=(sig + 1) / 2)
            noise = np.random.normal(0, 0.1, size=t.shape)
            inp = torch.tensor(pwm + noise).float()
        elif u == "pwm_sqr":
            sig = torch.sin(t)
            pwm = square(2 * t / 3, duty=(sig + 1) / 2)
            inp = torch.tensor(pwm).float()
        elif u == "saw":
            inp = torch.tensor(sawtooth(t, 1)).float()
        elif u == "chirp":
            inp = torch.tensor(chirp(t, f0=1e-3, f1=0.5, t1=30, method='logarithmic')).float()
        elif u == "brownian":
            normal = torch.randn_like(t)
            inp = torch.cumsum(normal, dim=0)
        else:
            raise NotImplementedError
        return inp

    def generate_data(self, y0, t_start, t_end, dt, u=None):
        t = torch.arange(t_start, t_end, dt)

        if self.solver in ("euler", "midpoint", "rk3", "rk4"):
            options = {"step_size": dt}
        elif self.solver == "dopri5":
            options = {}
        else:
            raise NotImplementedError

        if u is None:
            with torch.no_grad():
                y = odeint(self, y0, t, method=self.solver, options=options)
            y = y.squeeze()
            u = torch.zeros_like(y)
        else:  # If the model is controlled, the inputs need to be computed
            if type(u) in [list, tuple]:
                # If the model has multiple different inputs, they all need to be specified
                assert len(u) == self.n_inputs, "Specified number of inputs are different from model capacity"
                inp = torch.stack([self.generate_input(t, ui) for ui in u], dim=0)
                t_inp = t[None, :].expand(inp.shape[0], -1)
            else:
                if self.n_inputs > 1:
                    # If the model has multiple unspecified inputs, they are copied
                    inp = torch.stack([self.generate_input(t, u) for _ in range(self.n_inputs)], dim=0)
                    t_inp = t[None, :].expand(inp.shape[0], -1)
                else:
                    inp = self.generate_input(t, u)
                    t_inp = t

            # Create interpolation function to get intermediate inputs
            self._rgi = lambda ti: interp1d(t_inp, inp, ti[None])

            with torch.no_grad():
                y = odeint(self, y0, t, method=self.solver, options=options)
            y = y.squeeze()

            assert not torch.isnan(y).any()

            u = inp.T if inp.dim() > 1 else inp[:, None]

            self.reset_rgi()

        return y, t, u


class TwoStateLinear(ReferenceSystem):
    """The reference ODE to learn. Used to generate training data."""
    A = torch.tensor([[-3, 5.], [-5, -3]])  # eigenvalues at interesting places
    B = torch.tensor([[0., 1.]])

    def __init__(self, n_inputs=1):
        super().__init__(n_inputs=n_inputs)

    def forward(self, t, y):
        inp = self._rgi(t)
        dx = torch.mm(y, self.A) + self.B * inp
        return dx


class LinearMultiStateTeacherNetwork(ReferenceSystem):
    n_states: int = 2
    n_inputs: int = 1
    n_hidden: int = 16
    n_layers: int = 2
    solver: str = "dopri5"
    solver_order: int = 1
    exclusion_order: int = 0
    complex_poles: bool = False
    poles = None
    eps: float = -1e-2
    input_const: float = 0.1
    bias_init: float = 1e-4
    stochastic_inp: bool = False

    def __init__(self, config: dict):
        super().__init__()
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.A = nn.Sequential(nn.Linear(self.n_states, self.n_states, bias=False), )
        self.B = nn.Parameter(torch.randn(1, self.n_states) * self.input_const, requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        poles = init_strategy(self.A,
                              self.n_states,
                              0,
                              self.step_size,
                              self.solver_order,
                              self.complex_poles,
                              self.eps,
                              self.exclusion_order,
                              self.bias_init)
        self.poles = poles

    def state_transition(self, x, u):
        du = 0
        if self.n_inputs > 0:
            du = self.B * u
        return self.A(x) + du

    def state_mat(self):
        return self.A[0].weight.data

    def forward(self, t, y):
        du = 0
        if self.n_inputs > 0:
            du = self.B * self._rgi(t).view(y.shape[0], self.n_inputs)
        return self.A(y) + du


class MultiStateTeacherNetwork(ReferenceSystem):
    n_states: int = 2
    n_inputs: int = 1
    n_hidden: int = 16
    n_layers: int = 2
    step_size: float = 0.1
    solver: str = "dopri5"
    solver_order: int = 1
    exclusion_order: int = 0
    complex_poles: bool = False
    eps: float = -1e-2
    bias_init: float = 1e-4
    stochastic_inp: bool = False
    poles = None

    def __init__(self, config: dict, seed=None):
        super().__init__()
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.f = self._create_net(self.n_states, self.n_inputs, self.n_hidden, self.n_layers)
        if seed is not None:
            self.f.apply(lambda m: self.init_weights(m, seed))
        else:
            self.reset_parameters(self.f)

    def init_weights(self, m, seed):
        # Save the current state of the random number generator
        rng_state = torch.get_rng_state()

        # Set the seed
        torch.manual_seed(seed)

        if isinstance(m, nn.Sequential):
            self.reset_parameters(m)

        # Restore the state of the random number generator
        torch.set_rng_state(rng_state)

    def reset_parameters(self, m):
        poles = init_strategy(m,
                              self.n_states,
                              self.n_inputs,
                              self.step_size,
                              self.solver_order,
                              self.complex_poles,
                              self.eps,
                              self.exclusion_order,
                              self.bias_init,
                              self.stochastic_inp)
        self.poles = poles

    @staticmethod
    def _create_net(n_states, n_inputs, n_hidden, n_layers):
        def block(in_feat, out_feat, normalize=False, dropout=0.0):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.LayerNorm(out_feat))
            if dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ELU(inplace=True))
            return layers

        net = nn.Sequential(
            # input layer
            *block(n_states + n_inputs, n_hidden),

            # hidden layers
            *[module for _ in range(n_layers - 1) for module in block(n_hidden, n_hidden)],

            # output layer
            nn.Linear(n_hidden, n_states),
        )

        return net

    def state_transition(self, x, u):
        if self.n_inputs > 0:
            x = torch.cat((x, u), dim=-1)
        return self.f(x)

    @torch.inference_mode(False)
    def state_jacobian(self, x, u=None):
        batch_size, n_states = x.shape
        if u is None:
            u = torch.zeros_like(x)
        # calculates the Jacobian of the state transition function w.r.t. the states X
        jacobian = func.vmap(func.jacrev(self.state_transition, argnums=0))(x, u)
        F = jacobian.view(batch_size, n_states, n_states)
        return F

    def forward(self, t, y):
        if self.n_inputs > 0:
            u = self._rgi(t).view(y.shape[0], self.n_inputs)
            y = torch.cat((y, u), dim=-1)
        return self.f(y)
