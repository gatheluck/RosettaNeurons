from __future__ import annotations

import torch


def pearson_corr_activations(
    activation_map_1: torch.Tensor,
    activation_map_2: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute Pearson correlation between channels of two activation maps.

    The two activation maps are expected to have shape
    ``(batch, channels, height, width)``. The function treats
    ``batch * height * width`` as the sample dimension and computes
    the Pearson correlation coefficient between every pair of
    channels from ``activation_map_1`` and ``activation_map_2``.

    The computation is fully differentiable and runs on whatever
    device the input tensors are on (CPU or GPU).

    Args:
        activation_map_1: A 4-D tensor of shape
            ``(batch, channels_1, height, width)``.
        activation_map_2: A 4-D tensor of shape
            ``(batch, channels_2, height, width)``.
        eps: Small positive value used to avoid division by zero
            when normalizing each channel.

    Returns:
        A 2-D tensor of shape ``(channels_1, channels_2)`` containing
        Pearson correlation coefficients between channels of
        ``activation_map_1`` and channels of ``activation_map_2``.

    Raises:
        ValueError: If the input tensors do not have 4 dimensions,
            if their batch size or spatial dimensions do not match,
            or if they are not on the same device.
    """
    if activation_map_1.ndim != 4 or activation_map_2.ndim != 4:
        raise ValueError(
            "Both activation maps must be 4-D tensors of shape "
            "(batch, channels, height, width)."
        )

    if activation_map_1.shape[0] != activation_map_2.shape[0]:
        raise ValueError(
            "Batch dimension of both activation maps must match: "
            f"{activation_map_1.shape[0]} != {activation_map_2.shape[0]}"
        )

    if activation_map_1.shape[2:] != activation_map_2.shape[2:]:
        raise ValueError(
            "Spatial dimensions (height, width) of both activation maps "
            f"must match: {activation_map_1.shape[2:]} != {activation_map_2.shape[2:]}"
        )

    if activation_map_1.device != activation_map_2.device:
        raise ValueError(
            "Both activation maps must be on the same device "
            f"(got {activation_map_1.device} and {activation_map_2.device})."
        )

    # Use float32 for computation and keep the original device.
    activation_map_1 = activation_map_1.float()
    activation_map_2 = activation_map_2.float()

    batch_size, channels_1, height, width = activation_map_1.shape
    channels_2: int = activation_map_2.shape[1]
    num_samples: int = batch_size * height * width

    # Reshape to (channels, num_samples); each row is one flattened channel.
    activations_1: torch.Tensor = activation_map_1.permute(1, 0, 2, 3).reshape(
        channels_1, num_samples
    )
    activations_2: torch.Tensor = activation_map_2.permute(1, 0, 2, 3).reshape(
        channels_2, num_samples
    )

    # Center each channel by subtracting its mean.
    activations_1 = activations_1 - activations_1.mean(dim=1, keepdim=True)
    activations_2 = activations_2 - activations_2.mean(dim=1, keepdim=True)

    # Compute L2 norms per channel and avoid division by zero.
    norms_1: torch.Tensor = activations_1.norm(dim=1, keepdim=True).clamp_min(eps)
    norms_2: torch.Tensor = activations_2.norm(dim=1, keepdim=True).clamp_min(eps)

    # Normalize each channel to unit norm.
    normalized_1: torch.Tensor = activations_1 / norms_1
    normalized_2: torch.Tensor = activations_2 / norms_2

    # Compute correlation matrix as matrix multiplication (channels_1 x channels_2).
    correlation_matrix: torch.Tensor = normalized_1 @ normalized_2.T

    # Clamp to valid range and replace NaNs with zero.
    correlation_matrix = correlation_matrix.clamp(-1.0, 1.0)
    return torch.nan_to_num(correlation_matrix, nan=0.0)
