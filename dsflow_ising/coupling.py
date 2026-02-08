"""Discrete coupling layers for ±1 spin configurations.

Each coupling layer:
1. Splits spins into A and B sublattices (checkerboard partition)
2. Passes z_A through a small ConvNet (MaskNet) to produce signs for B sites
3. Multiplies z_B by those signs: σ_B = z_B * sgn(g_φ(z_A))
4. Leaves A unchanged: σ_A = z_A

The transform is self-inverse: applying the same layer again recovers z.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence


def checkerboard_indices(L: int, partition: str):
    """Return indices of A and B sublattice sites.

    Args:
        L: lattice side length
        partition: "even" or "odd"
            "even": A-sites have (i+j) % 2 == 0, B-sites have (i+j) % 2 == 1
            "odd":  A-sites have (i+j) % 2 == 1, B-sites have (i+j) % 2 == 0

    Returns:
        (a_idx, b_idx): index arrays into the flattened (L*L,) spin array
    """
    a_list, b_list = [], []
    parity = 0 if partition == "even" else 1
    for i in range(L):
        for j in range(L):
            site = i * L + j
            if (i + j) % 2 == parity:
                a_list.append(site)
            else:
                b_list.append(site)
    return jnp.array(a_list, dtype=jnp.int32), jnp.array(b_list, dtype=jnp.int32)


class MaskNet(nn.Module):
    """Small ConvNet that maps A-sublattice values to pre-sign logits for B-sites.

    Input: A-sublattice values placed on L×L grid (B-sites zeroed out).
    Output: logits for each site (only B-site values are used).
    """
    L: int
    features: Sequence[int] = (16, 16)

    @nn.compact
    def __call__(self, z_grid):
        """
        Args:
            z_grid: shape (L, L) or (batch, L, L) — full grid with B-sites zeroed.

        Returns:
            logits: shape same as input — logits for all sites.
        """
        # Ensure 4D for Conv: (batch, H, W, C)
        orig_shape = z_grid.shape
        if z_grid.ndim == 2:
            x = z_grid[None, :, :, None]  # (1, L, L, 1)
        elif z_grid.ndim == 3:
            x = z_grid[:, :, :, None]  # (batch, L, L, 1)
        else:
            x = z_grid

        for feat in self.features:
            x = nn.Conv(feat, kernel_size=(3, 3), padding='SAME')(x)
            x = nn.relu(x)
        x = nn.Conv(1, kernel_size=(3, 3), padding='SAME')(x)  # (batch, L, L, 1)
        x = x.squeeze(-1)  # (batch, L, L)

        if len(orig_shape) == 2:
            x = x.squeeze(0)  # (L, L)
        return x


def _binary_sign(x):
    """Binary sign: x >= 0 → +1, x < 0 → -1. Always in {-1, +1}."""
    return jnp.where(x >= 0, 1.0, -1.0)


def _ste_sign(x):
    """Sign function with straight-through estimator for gradients.

    Forward: returns binary sign(x) ∈ {-1, +1} (never 0)
    Backward: gradient passes through as if identity (∂sign/∂x = 1)
    """
    return x + jax.lax.stop_gradient(_binary_sign(x) - x)


def forward_layer(mask_net, params, z, L, partition, use_ste=True):
    """Apply one discrete coupling layer.

    Args:
        mask_net: MaskNet instance
        params: MaskNet parameters
        z: spin config in {-1, +1}, shape (..., N) where N = L*L
        L: lattice side length
        partition: "even" or "odd"
        use_ste: if True, use straight-through estimator for sign

    Returns:
        sigma: transformed spin config, shape (..., N)
    """
    N = L * L
    a_idx, b_idx = checkerboard_indices(L, partition)

    batch_shape = z.shape[:-1]
    z_flat = z.reshape(-1, N) if batch_shape else z[None]  # (B, N)
    B = z_flat.shape[0]

    # Build grid with only A-sublattice values; B-sites zeroed
    z_grid = jnp.zeros((B, L, L))
    z_grid = z_grid.at[:, a_idx // L, a_idx % L].set(z_flat[:, a_idx])

    # Get logits from mask network
    logits = mask_net.apply(params, z_grid)  # (B, L, L)
    logits_flat = logits.reshape(B, N)
    logits_b = logits_flat[:, b_idx]  # (B, n_B)

    # Compute signs
    sign_fn = _ste_sign if use_ste else _binary_sign
    m = sign_fn(logits_b)  # (B, n_B) in {-1, +1}

    # Apply: σ_B = z_B * m, σ_A = z_A
    sigma_flat = z_flat.at[:, b_idx].set(z_flat[:, b_idx] * m)

    if batch_shape:
        return sigma_flat.reshape(*batch_shape, N)
    else:
        return sigma_flat.squeeze(0)


def inverse_layer(mask_net, params, sigma, L, partition, use_ste=True):
    """Inverse of the coupling layer (identical to forward — it's self-inverse)."""
    return forward_layer(mask_net, params, sigma, L, partition, use_ste=use_ste)
