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

import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.container import Sequential
from torch.nn.modules.linear import Linear


def sample_poles(n_states: int,
                 step_size: float = 0.1,
                 solver_order: int = 1,
                 use_imag: bool = False,
                 eps: float = -1e-1,
                 exclusion_order: int = 0
                 ) -> torch.Tensor:
    """
    Generate "n_states" number of poles using rejection sampling based
     on the stability region of the solver of order "solver_order"

    Parameters
    ----------
    n_states : int
        The number of dynamic states.
    step_size : float
        The solver step size.
    solver_order : int
        The order of the solver, e.g. Euler forward is a first-order solver.
        Should be within (1, 2, 3, 4).
    use_imag : bool
        If imaginary poles should be included. Default is false.
    eps : float
        (Small) value to make sure that samples are not
         on the border of the stability regions
    exclusion_order : int
        The order of the solver region to not include.

    Returns
    -------
    poles: torch.Tensor
        a torch tensor of poles with size: (n_states)

    Raises
    ------
    AssertionError
        - If the solver order is not in (1, 2, 3, 4)
        - If the exclusion order is not smaller the solver order

    """

    assert solver_order in (1, 2, 3, 4), "The solver order should be within [1, 2, 3, 4]"

    real_range = [-3.0, -0.1]
    im_range = [-3.0, 3.0]

    x: list[float] = []
    y: list[float] = []

    while len(x) < n_states:
        xi = (real_range[0] - real_range[-1]) * np.random.uniform() + real_range[-1]
        yi = (im_range[0] - im_range[-1]) * np.random.uniform() + im_range[-1]

        # if not use_imag:
        #    yi = 0.

        if (n_states % 2 != 0 and len(x) == n_states - 1) or not use_imag:
            yi = 0.

        z = xi + 1j * yi
        region = 1.0 + 0.0j
        for p in range(1, solver_order + 1):
            region += (z ** p) / math.factorial(p)

        exclusion_region = 1.0 + 0.0j
        if exclusion_order > 0:
            assert exclusion_order < solver_order
            for p in range(1, exclusion_order + 1):
                exclusion_region += (z ** p) / math.factorial(p)

        if np.abs(region) < 1 + eps:
            if exclusion_order > 0:
                if np.abs(exclusion_region) < 1 - eps:
                    continue
                else:
                    x.append(xi)
                    y.append(yi)
            else:
                x.append(xi)
                y.append(yi)

    x_tensor = torch.tensor(x) / step_size
    y_tensor = torch.tensor(y) / step_size

    if use_imag:
        # If imaginary poles are included, we make sure they are conjugate pairs.
        # This is to make sure the reference "A"-matrix is real-valued.
        poles = x_tensor.to(torch.complex64)
        j = 0
        N = len(x_tensor)
        if N % 2 != 0:
            N -= 1

        while j < N:
            xj = x_tensor[j]
            yj = y_tensor[j]
            try:
                poles[j] = xj + 1j * yj
                poles[j + 1] = xj - 1j * yj
            except IndexError:
                break
            j += 2
    else:
        poles = x_tensor.to(torch.complex64)  # For consistency
    return poles


def sample_exclusive_poles(n_states: int,
                           step_size: float = 0.1,
                           use_imag: bool = False,
                           eps: float = -1e-1,
                           exclusion_order: int = 0
                           ) -> torch.Tensor:
    """
    Generate "n_states" number of poles using rejection sampling based
     on the stability region of the solver not to include (exclusion_order)
     if exclusion_order == 0, than any poles within the specified ranges may be included

    Parameters
    ----------
    n_states : int
        The number of dynamic states.
    step_size : float
        The solver step size.
    use_imag : bool
        If imaginary poles should be included. Default is false.
    eps : float
        (Small) value to make sure that samples are not
         on the border of the stability regions
    exclusion_order : int
        The order of the solver region to not include.

    Returns
    -------
    poles: torch.Tensor
        a torch tensor of poles with size: (n_states)

    """

    real_range = [-3.0, -0.1]
    im_range = [-3.0, 3.0]

    x: list[float] = []
    y: list[float] = []

    while len(x) < n_states:
        xi = (real_range[0] - real_range[-1]) * np.random.uniform() + real_range[-1]
        yi = (im_range[0] - im_range[-1]) * np.random.uniform() + im_range[-1]

        z = xi + 1j * yi

        exclusion_region = 1.0 + 0.0j
        for p in range(1, exclusion_order + 1):
            exclusion_region += (z ** p) / math.factorial(p)

        if np.abs(exclusion_region) < 1 - eps:
            continue
        else:
            x.append(xi)
            y.append(yi)

    x_tensor = torch.tensor(x) / step_size
    y_tensor = torch.tensor(y) / step_size

    if use_imag:
        # If imaginary poles are included, we make sure they are conjugate pairs.
        # This is to make sure the reference "A"-matrix is real-valued.
        poles = x_tensor.to(torch.complex64)
        j = 0
        N = len(x_tensor)
        if N % 2 != 0:
            N -= 1

        while j < N:
            xj = x_tensor[j]
            yj = y_tensor[j]
            try:
                poles[j] = xj + 1j * yj
                poles[j + 1] = xj - 1j * yj
            except IndexError:
                break
            j += 2
    else:
        poles = x_tensor.to(torch.complex64)  # For consistency
    return poles


def ortho_matrix(n: int,
                 use_imag: bool = False
                 ) -> torch.Tensor:
    """
    Generate a random orthogonal matrix, drawn from the O(n) Haar distribution

    Parameters
    ----------
    n : int
        The dimension of the orthogonal matrix
    use_imag : bool, optional
        If True, include complex entries in the generated matrix. Default is False.

    Returns
    -------
    orthogonal_matrix : torch.Tensor
        A torch tensor representing the generated orthogonal matrix of size (n, n).

    """

    # Generate a random matrix with or without imaginary entries
    if use_imag:
        random_matrix = torch.randn(n, n) + 1j * torch.randn(n, n) / math.sqrt(2.0)
    else:
        random_matrix = torch.randn(n, n)

    # Perform QR decomposition of the random matrix
    q, r = torch.linalg.qr(random_matrix)

    # Normalize the diagonal of the R matrix
    diagonal = r.diagonal(offset=0, dim1=-2, dim2=-1)
    phase = diagonal / diagonal.abs()

    # Multiply Q by the normalized diagonal to obtain the orthogonal matrix
    orthogonal_matrix = q * phase

    return orthogonal_matrix


def init_strategy(seq_module: Sequential,
                  n_states: int,
                  n_inputs: int,
                  step_size: float = 0.1,
                  solver_order: int = 1,
                  use_imag: bool = False,
                  eps: float = -1e-2,
                  exclusion_order: int = 0,
                  bias_init: float = 1e-4,
                  stochastic_inp: bool = False
                  ) -> torch.Tensor:
    """
    Initialization strategy for the given sequential module.

    Builds upon the assumption that the dynamic model
        dx = f(x, u)
    can be approximately described by
        dx = Ax + Bu
    where the matrix A is referred to as the reference "A"-matrix.

    Parameters
    ----------
    seq_module : torch.nn.Sequential
        The input sequential module to be processed.
    n_states : int
        The number of states in the model.
    n_inputs : int
        The number of inputs in the model.
    step_size : float, optional
        Some parameter (default is 0.1).
    solver_order : int, optional
        The order of the solver (default is 1).
    use_imag : bool, optional
        If True, include complex poles. Default is False.
    eps : float, optional
        A small constant (default is 1e-2).
    exclusion_order : int optional
        The order of the solver region to not include (default is 0).
    bias_init : float, optional
        The initial bias value (default is 1e-4).
    stochastic_inp : bool, optional
        Whether some states should have inputs or not.

    Returns
    ----------
    poles : torch.tensor
        tensor of reference poles

    Raises
    ------
    AssertionError
        If the input is not of the expected type or the number of outputs does not match the number of states.

    """

    # Check that the input is of the expected type
    assert isinstance(seq_module, Sequential), "The first input should be of type nn.Sequential"

    # Extract the weights of the linear layers in the sequential module
    linear_weights = [module.weight.data for module in seq_module if isinstance(module, Linear)]

    # Check that the number of outputs matches the number of states
    assert n_states == linear_weights[-1].shape[0], "The number of outputs should be the same as the number of states"

    # Calculate the number of hidden units and layers
    n_hidden, _ = linear_weights[0].shape
    n_layers = len(linear_weights)

    # Sample poles for the solver stability region
    if solver_order > 4:
        poles = sample_exclusive_poles(n_states, step_size, use_imag=use_imag,
                                       eps=eps, exclusion_order=exclusion_order)
    else:
        poles = sample_poles(n_states, step_size, solver_order, use_imag=use_imag,
                             eps=eps, exclusion_order=exclusion_order)

    # Create reference "A"-matrix
    A = torch.zeros(n_states, n_states)
    j = 0

    while j < n_states:
        pole = poles[j]
        if pole.imag.abs() > 0:

            pole = (-1 * pole) ** (1 / n_layers)

            mu = pole.real
            omega = pole.imag

            A[j, j] = A[j + 1, j + 1] = mu
            A[j, j + 1] = omega
            A[j + 1, j] = omega.neg()

            j += 2
        else:
            A[j, j] = (-1 * pole.real) ** (1 / n_layers)
            j += 1

    if n_layers == 1:
        for module in seq_module:
            if isinstance(module, Linear):
                module.weight.data = -A.T
                return poles
    ji = 0
    new_weights = []

    past = n_states + n_inputs
    nxt = n_hidden

    t_pst = torch.eye(past)
    t_nxt = ortho_matrix(nxt)

    for module in seq_module:
        if isinstance(module, Linear):

            cn = torch.zeros(past, nxt)

            # if past <= nxt:
            #    cn[:past, :past] = torch.eye(past)
            # else:
            #    cn[:nxt, :nxt] = torch.eye(nxt)

            cn[:n_states, :n_states] = A
            if ji == 0:
                bound = A.diagonal().abs().mean().item()
                # bound = 1 / math.sqrt(past)
                input_init = torch.rand(n_inputs, n_states).uniform_(-bound, bound)

                if stochastic_inp:
                    # Assure that the reference the teacher is different from the student
                    input_init = torch.rand(n_inputs, n_states).normal_(0, bound)
                    input_prob = torch.zeros_like(input_init).bernoulli_()
                    while (input_prob == 0).all():
                        input_prob.bernoulli_()
                    input_init *= input_prob

                cn[n_states:n_states + n_inputs, :n_states] = input_init

            weight = t_pst.T @ cn
            # weight = cn

            if ji == 0:
                weight = (-1) * weight @ t_nxt
            elif ji < n_layers - 1:
                weight = weight @ t_nxt

            if module.bias is not None:
                # bound = 1 / math.sqrt(past)
                nn.init.uniform_(module.bias, (-1) * bias_init, bias_init)
                # nn.init.uniform_(module.bias, (-1) * bound, bound)

            module.weight.data = weight.T

            new_weights.append(weight)

            t_pst = t_nxt
            past = nxt

            try:
                nxt = linear_weights[ji + 1].shape[0]
            except IndexError:
                break
            else:
                ji += 1
                t_nxt = ortho_matrix(nxt)

    return poles
