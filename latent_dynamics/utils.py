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


def pixel_accuracy(pred: torch.Tensor,
                   target: torch.Tensor,
                   threshold: float = 0.025
                   ) -> float:
    """
    Compute pixel accuracy for image regression tasks with temporal dimension.

    Args:
    - pred (torch.Tensor): Predicted tensor of shape (batch_size, time_steps, channels, height, width).
    - target (torch.Tensor): Target tensor of shape (batch_size, time_steps, channels, height, width).
    - threshold (float): Threshold for acceptable pixel difference.

    Returns:
    - accuracy (float): Proportion of pixels below the threshold difference.
    """

    # Compute the absolute difference between predicted and target tensors
    diff = torch.abs(pred - target)

    # Count pixels with difference below the threshold
    accurate_pixels = torch.sum(diff <= threshold)

    # Total number of pixels
    total_pixels = torch.numel(pred)

    accuracy = accurate_pixels.float() / total_pixels

    return accuracy.item()


def weighted_pixel_accuracy(pred: torch.Tensor,
                            target: torch.Tensor,
                            threshold: float = 0.025
                            ) -> float:
    """
    Compute weighted pixel accuracy, dynamically adjusting for black pixel prevalence.

    Args:
    - pred (torch.Tensor): Predicted image tensor of shape (batch_size, channels, height, width).
    - target (torch.Tensor): Target image tensor of shape (batch_size, channels, height, width).
    - threshold (float): Threshold for defining black pixels.

    Returns:
    - accuracy (float): Weighted proportion of pixels below the threshold difference.
    """

    # Compute the absolute difference between predicted and target images
    diff = torch.abs(pred - target)

    # Mask for pixels that are gray
    black_mask = target < 0.37

    # Compute fraction of black pixels
    black_fraction = torch.sum(black_mask).float() / torch.numel(target)

    # Compute weights for black pixels (inverse of their prevalence)
    black_pixel_weight = 1.0 / (black_fraction + 1e-10)

    # Assign weights based on black pixel prevalence
    weights = torch.where(black_mask, black_pixel_weight, 1.0)

    # Compute weighted accurate pixels count
    accurate_pixels = torch.sum((diff <= threshold).float() * weights)

    # Total weight (for normalization)
    total_weight = torch.sum(weights)

    accuracy = accurate_pixels / total_weight
    return accuracy.item()


def performance_decay(pred: torch.Tensor,
                      target: torch.Tensor
                      ) -> float:
    """
    Compute the performance decay for a sequence of predicted images.

    Args:
    - pred (torch.Tensor): Predicted tensor of shape (batch_size, time_steps, channels, height, width).
    - target (torch.Tensor): Target tensor of shape (batch_size, time_steps, channels, height, width).

    Returns:
    - decay (float): Performance decay.
    """

    # Compute the MSE between predicted and target tensors
    diff = (pred - target) ** 2
    y = torch.mean(diff, dim=(0, 2, 3, 4))
    x = torch.arange(len(y), dtype=torch.float32, device=y.device)

    mu_x = torch.mean(x)
    mu_y = torch.mean(y)

    k = torch.sum((x - mu_x) * (y - mu_y)) / (torch.sum((x - mu_x) ** 2) + 1e-10)

    return k.item()


def valid_prediction_time(pred: torch.Tensor,
                          target: torch.Tensor,
                          threshold: float = 0.025
                          ) -> float:
    """
    Compute the smallest time steps where the MSE is above a threshold.

    Args:
    - pred (torch.Tensor): Predicted tensor of shape (batch_size, time_steps, channels, height, width).
    - target (torch.Tensor): Target tensor of shape (batch_size, time_steps, channels, height, width).
    - threshold (float): Threshold for acceptable pixel difference.

    Returns:
    - time: the time index that minimizes the condition
    """

    # Compute the MSE between predicted and target tensors
    diff = (pred - target) ** 2
    mse = torch.mean(diff, dim=(2, 3, 4))

    # Convert boolean tensor to float tensor
    above_threshold = torch.where(mse > threshold, 1., 0.)

    # Fix the case when all mse are below threshold
    above_threshold[:, -1] = torch.where(above_threshold[:, -1] == 0., 1., 1.)

    # Find the first time step where the MSE is above the threshold
    time = torch.argmax(above_threshold, dim=1)

    # Average of these times:
    avg_time = time.float().mean() / pred.shape[1]

    return avg_time.item()
