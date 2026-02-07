"""Ising model energy and lattice utilities for 2D periodic square lattice."""

import jax.numpy as jnp


def nearest_neighbor_pairs(L: int) -> jnp.ndarray:
    """Return (num_pairs, 2) index array for L×L periodic square lattice.

    Each bond appears exactly once. Total bonds = 2*L*L (horizontal + vertical).
    Sites are indexed in row-major order: site = i*L + j.
    """
    pairs = []
    for i in range(L):
        for j in range(L):
            site = i * L + j
            # Right neighbor (periodic)
            right = i * L + (j + 1) % L
            pairs.append((site, right))
            # Down neighbor (periodic)
            down = ((i + 1) % L) * L + j
            pairs.append((site, down))
    return jnp.array(pairs, dtype=jnp.int32)


def energy(sigma: jnp.ndarray, pairs: jnp.ndarray, J: float = 1.0) -> jnp.ndarray:
    """Compute Ising energy E(σ) = -J Σ_{<ij>} σ_i σ_j.

    Args:
        sigma: spin configurations, shape (..., N) with values in {-1, +1}
        pairs: (num_pairs, 2) bond index array
        J: coupling constant

    Returns:
        Energy per configuration, shape (...)
    """
    si = sigma[..., pairs[:, 0]]  # (..., num_pairs)
    sj = sigma[..., pairs[:, 1]]  # (..., num_pairs)
    return -J * jnp.sum(si * sj, axis=-1)


def magnetization(sigma: jnp.ndarray) -> jnp.ndarray:
    """Compute absolute magnetization |⟨σ⟩| per sample.

    Args:
        sigma: spin configurations, shape (..., N)

    Returns:
        |m| per configuration, shape (...)
    """
    return jnp.abs(jnp.mean(sigma, axis=-1))
