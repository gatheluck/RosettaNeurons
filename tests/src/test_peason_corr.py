from __future__ import annotations

import pytest
import torch

import src.pearson_corr


def _pearson_corr_activations_naive(
    activation_map_1: torch.Tensor,
    activation_map_2: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Naive reference implementation of Pearson correlation for testing.

    Args:
        activation_map_1: First activation map of shape (B, C1, H, W).
        activation_map_2: Second activation map of shape (B, C2, H, W).
        eps: Small value to avoid division by zero.

    Returns:
        A tensor of shape (C1, C2) with Pearson correlation coefficients.
    """
    if activation_map_1.ndim != 4 or activation_map_2.ndim != 4:
        raise ValueError("Inputs must be 4-D tensors of shape (B, C, H, W).")

    if activation_map_1.shape[0] != activation_map_2.shape[0]:
        raise ValueError("Batch dimension of both activation maps must match.")

    if activation_map_1.shape[2:] != activation_map_2.shape[2:]:
        raise ValueError("Spatial dimensions of both activation maps must match.")

    batch_size, channels_1, height, width = activation_map_1.shape
    channels_2: int = activation_map_2.shape[1]
    num_samples: int = batch_size * height * width

    act1: torch.Tensor = (
        activation_map_1.float().permute(1, 0, 2, 3).reshape(channels_1, num_samples)
    )
    act2: torch.Tensor = (
        activation_map_2.float().permute(1, 0, 2, 3).reshape(channels_2, num_samples)
    )

    result: torch.Tensor = torch.empty(
        channels_1,
        channels_2,
        device=act1.device,
        dtype=act1.dtype,
    )

    for i in range(channels_1):
        x: torch.Tensor = act1[i]
        x = x - x.mean()
        norm_x: torch.Tensor = x.norm().clamp_min(eps)

        for j in range(channels_2):
            y: torch.Tensor = act2[j]
            y = y - y.mean()
            norm_y: torch.Tensor = y.norm().clamp_min(eps)

            denom: torch.Tensor = norm_x * norm_y
            if denom.item() <= eps:
                result[i, j] = 0.0
            else:
                result[i, j] = (x @ y) / denom

    result = result.clamp(-1.0, 1.0)
    return torch.nan_to_num(result, nan=0.0)


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Return CUDA device if available, otherwise CPU.

    All tests in this module will run on this device, so if a GPU is
    available they will exercise the CUDA path automatically.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_pearson_corr_matches_naive(device: torch.device) -> None:
    """pearson_corr_activations should match the naive implementation."""
    activation_map_1: torch.Tensor = torch.randn(2, 3, 4, 4, device=device)
    activation_map_2: torch.Tensor = torch.randn(2, 5, 4, 4, device=device)

    expected: torch.Tensor = _pearson_corr_activations_naive(
        activation_map_1, activation_map_2
    )
    actual: torch.Tensor = src.pearson_corr.pearson_corr_activations(
        activation_map_1, activation_map_2
    )

    assert actual.shape == expected.shape
    assert actual.dtype == torch.float32
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_pearson_corr_identical_maps_diagonal_one(device: torch.device) -> None:
    """Diagonal of correlation matrix should be close to 1 for identical maps."""
    activation_map: torch.Tensor = torch.randn(3, 4, 3, 3, device=device)

    corr: torch.Tensor = src.pearson_corr.pearson_corr_activations(
        activation_map, activation_map
    )

    diagonal: torch.Tensor = torch.diagonal(corr)
    assert torch.allclose(diagonal, torch.ones_like(diagonal), atol=1e-5)
    # Symmetry check when both inputs are the same.
    assert torch.allclose(corr, corr.T, atol=1e-5)


def test_pearson_corr_constant_channel_zero(device: torch.device) -> None:
    """Channels with zero variance should produce zero correlation."""
    # First activation map: second channel is constant.
    activation_map_1: torch.Tensor = torch.zeros(1, 2, 3, 3, device=device)
    activation_map_1[0, 1] = 5.0  # make channel 1 constant non-zero

    activation_map_2: torch.Tensor = torch.randn(1, 2, 3, 3, device=device)

    corr: torch.Tensor = src.pearson_corr.pearson_corr_activations(
        activation_map_1, activation_map_2
    )

    # Channel with zero variance should have zero correlation with all others.
    zero_row: torch.Tensor = corr[1]
    assert torch.allclose(zero_row, torch.zeros_like(zero_row), atol=1e-6)


def test_pearson_corr_raises_on_shape_mismatch(device: torch.device) -> None:
    """Function should raise ValueError on incompatible shapes."""
    activation_map_1: torch.Tensor = torch.randn(1, 2, 3, 3, device=device)
    # Different spatial size
    activation_map_2: torch.Tensor = torch.randn(1, 2, 4, 4, device=device)

    with pytest.raises(ValueError):
        _ = src.pearson_corr.pearson_corr_activations(
            activation_map_1, activation_map_2
        )
