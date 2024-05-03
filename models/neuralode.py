import warnings
import torch
import torch.nn as nn
from torch import func
from torchdiffeq import odeint
# from torchdiffeq import odeint_adjoint as odeint
from stb_init import init_strategy

solver_orders = {"euler": 1,
                 "midpoint": 2,
                 "adaptive_heun": 2,
                 "rk3": 3,
                 "rk4": 4,
                 "dopri5": 4}


class ParamNet(nn.Module):
    def __init__(self, n_states):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(n_states, n_states).uniform_(-1e-1, 1e-1), requires_grad=True)
        self.b = nn.Parameter(torch.ones(n_states, n_states) * 5, requires_grad=False)

    def forward(self, x):
        return x @ self.a + self.b


class LinearNet(nn.Module):
    def __init__(self, n_states, n_inputs=0, perturb_val=1.0):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(n_states, n_states), requires_grad=True)
        self.B = nn.Parameter(torch.zeros(n_inputs, n_states), requires_grad=(n_inputs > 0))
        self.controlled = (n_inputs > 0)
        self.reset_parameters(n_states, n_inputs, perturb_val)

    def reset_parameters(self, n_states, n_inputs, perturb_val=1.0):
        from math import sqrt
        std = 1 / sqrt(n_states) * perturb_val
        nn.init.uniform_(self.A.data, -std, std)
        if n_inputs > 0:
            nn.init.uniform_(self.B.data, -std, std)

    def get_poles(self):
        return torch.linalg.eigvals(self.A).detach()

    def forward(self, X):
        x, u = X
        dx = x @ self.A + u @ self.B
        return dx


class NonLinearNet(nn.Module):
    poles = None

    def __init__(self,
                 n_states: int,
                 n_inputs: int = 0,
                 n_hidden: int = 8,
                 n_layers: int = 1,
                 step_size: float = 1.0,
                 solver_order: int = 0,
                 complex_poles: bool = False,
                 exclusion_order: int = 0,
                 perturb_val: float = 1.0,
                 activation: str = "elu"):
        super().__init__()

        self.controlled = (n_inputs > 0)
        self.f = self._create_net(n_states, n_inputs, n_hidden, n_layers, activation)

        if solver_order > 0:
            self.poles = self._reset_parameters(n_states, n_inputs, step_size,
                                                solver_order, complex_poles, exclusion_order)
        else:
            self._reset_parameters_basic(n_states, n_inputs, perturb_val)

    def _reset_parameters_basic(self, n_states, n_inputs, perturb_val=1.0):
        from math import sqrt
        fan_in = n_states + n_inputs
        for p in self.parameters():
            if p.dim() > 1:
                fan_in = p.shape[-1]
            std = 1 / sqrt(fan_in) * perturb_val
            nn.init.uniform_(p, -std, std)

    def _reset_parameters(self, n_states, n_inputs, step_size=1.0, solver_order=1,
                          use_imag=False, exclusion_order=0):
        poles = init_strategy(self.f, n_states, n_inputs, step_size, solver_order,
                              use_imag, exclusion_order=exclusion_order)
        return poles

    @staticmethod
    def _create_net(n_states, n_inputs, n_hidden, n_layers, activation="elu"):
        def block(in_feat, out_feat, nonlinearity="elu", normalize=False, dropout=0.0):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.LayerNorm(out_feat))
            if dropout:
                layers.append(nn.Dropout(dropout))

            match nonlinearity:
                case "elu":
                    layers.append(nn.ELU(inplace=True))
                case "swish":
                    layers.append(nn.SiLU(inplace=True))
                case "gelu":
                    layers.append(nn.GELU(approximate="tanh"))
                case "relu":
                    layers.append(nn.ReLU(inplace=True))
                case "lrelu":
                    layers.append(nn.LeakyReLU(inplace=True))
                case "selu":
                    layers.append(nn.SELU(inplace=True))
                case "softplus":
                    layers.append(nn.Softplus())
                case "tanh":
                    layers.append(nn.Tanh())
                case "sigmoid":
                    layers.append(nn.Sigmoid())
                case "none":
                    # Will return a linear model
                    pass
                case _:
                    warnings.warn(f"Unknown activation function: {nonlinearity}. Using ELU.")
                    layers.append(nn.ELU(inplace=True))

            return layers

        net = nn.Sequential(
            # input layer
            *block(n_states + n_inputs, n_hidden, activation),

            # hidden layers
            *[module for _ in range(n_layers - 1) for module in block(n_hidden, n_hidden, activation)],

            # output layer
            nn.Linear(n_hidden, n_states),
        )

        return net

    def forward(self, X):
        x, u = X
        if self.controlled:
            x = torch.cat((x, u), dim=-1)
        return self.f(x)


class NeuralODE(nn.Module):
    _solver = 'euler'
    _h = 0.01
    _opts = {}
    _atol = 1e-9
    _rtol = 1e-7

    def __init__(self,
                 n_states: int,
                 n_inputs: int = 0,
                 n_hidden: int = 8,
                 n_layers: int = 1,
                 solver: str = 'euler',
                 h: float = 0.01,
                 perturb_val: float = 1.0,
                 linear: bool = False,
                 stability_init: bool = False,
                 complex_poles: bool = False,
                 activation: str = "elu"):
        super().__init__()
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.solver = solver
        self.step_size = h
        if linear:
            self.f = LinearNet(n_states, n_inputs, perturb_val)
        else:
            solver_order = solver_orders[solver] if stability_init else 0
            self.f = NonLinearNet(n_states,
                                  n_inputs,
                                  n_hidden,
                                  n_layers,
                                  h,
                                  solver_order,
                                  complex_poles,
                                  perturb_val=perturb_val,
                                  activation=activation)

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, method):
        assert method in ('euler', 'midpoint', 'rk3', 'rk4', 'adaptive_heun', 'dopri5')
        self._solver = method
        if method in ('adaptive_heun', 'dopri5'):
            self._opts = {}

    @property
    def opts(self):
        return self._opts

    @opts.setter
    def opts(self, options):
        assert options is dict
        self._opts = options

    @property
    def step_size(self):
        return self._h

    @step_size.setter
    def step_size(self, h):
        self._h = h
        if self._solver in ('euler', 'midpoint', 'rk3', 'rk4'):
            self._opts = {'step_size': self._h}
        else:
            self._opts = {}

    @property
    def atol(self):
        return self._atol

    @atol.setter
    def atol(self, atol):
        self._atol = atol

    @property
    def rtol(self):
        return self._rtol

    @rtol.setter
    def rtol(self, rtol):
        self._rtol = rtol

    def model_update(self, t, X):
        # Continuous function
        dx = self.f(X)
        du = torch.zeros_like(X[-1])
        return dx, du

    def state_transition(self, x, u):
        return self.f((x, u))

    @torch.inference_mode(False)
    def state_jacobian(self, X, inp=None):
        N, feat_dim = X.shape
        if inp is None:
            inp = torch.zeros_like(X)
        # calculates the Jacobian of the state transition function w.r.t. the states X
        jacobian = func.vmap(func.jacrev(self.state_transition, argnums=0))(X, inp)
        F = jacobian.view(N, feat_dim, feat_dim)
        return F

    def forward(self, t, x, u=None):
        # Discrete update
        if u is None:
            u = torch.zeros_like(x)

        x_next, _ = odeint(self.model_update, (x, u), t=t,
                           rtol=self._rtol, atol=self._atol,
                           method=self._solver, options=self._opts)

        return x_next

    def l2_norm(self):
        l2_norm = 0
        for W in self.parameters():
            l2_norm += torch.norm(W, 2).sum()
        return l2_norm

    def l1_norm(self):
        l1_norm = 0
        for W in self.parameters():
            l1_norm += torch.abs(W).sum()
        return l1_norm
