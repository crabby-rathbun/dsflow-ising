"""MADE (Masked Autoencoder for Distribution Estimation) for binary spins.

Implements an autoregressive model over ±1 spins using raster-scan ordering.
Each conditional p(z_k | z_{<k}) is a Bernoulli parameterized by masked dense layers.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence


class MaskedDense(nn.Module):
    """Dense layer with a fixed binary mask applied to the weight matrix."""
    features: int
    mask: jnp.ndarray  # binary mask, shape (in_features, features)

    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            'kernel',
            nn.initializers.lecun_normal(),
            (x.shape[-1], self.features),
        )
        bias = self.param('bias', nn.initializers.zeros, (self.features,))
        return x @ (kernel * self.mask) + bias


class MADE(nn.Module):
    """Masked Autoencoder for Distribution Estimation over ±1 spins.

    Uses raster-scan ordering: z_0, z_1, ..., z_{N-1}.
    Output logit_k depends only on z_0, ..., z_{k-1}.
    logit_k > 0 means p(z_k = +1 | z_{<k}) > 0.5.
    """
    n_sites: int
    hidden_dims: Sequence[int] = ()

    def setup(self):
        N = self.n_sites
        hdims = self.hidden_dims if self.hidden_dims else (4 * N,)

        # Assign ordering indices.
        # Input units get orders 0..N-1 (matching the spin index).
        # Hidden units get orders distributed across 0..N-2 so that
        # each hidden unit is allowed to depend on some subset of inputs.
        # Output unit k needs to depend on inputs < k, so it connects to
        # hidden units with order < k, i.e. hidden order <= k-1.

        # Build masks
        input_order = jnp.arange(N)  # (N,)

        masks = []
        prev_order = input_order
        prev_dim = N

        for h_dim in hdims:
            # Hidden units get evenly spaced orders in [0, N-2]
            hidden_order = jnp.arange(h_dim) % (N - 1) if N > 1 else jnp.zeros(h_dim, dtype=jnp.int32)
            # Connection: hidden unit h connects to prev unit p iff prev_order[p] <= hidden_order[h]
            mask = (prev_order[:, None] <= hidden_order[None, :]).astype(jnp.float32)
            masks.append(mask)
            prev_order = hidden_order
            prev_dim = h_dim

        # Output mask: output k connects to hidden h iff hidden_order[h] <= k-1, i.e. < k
        output_order = jnp.arange(N)
        output_mask = (prev_order[:, None] < output_order[None, :]).astype(jnp.float32)
        masks.append(output_mask)

        self.layers = [MaskedDense(features=m.shape[1], mask=m) for m in masks]

    def __call__(self, z):
        """Compute logits for each conditional p(z_k | z_{<k}).

        Args:
            z: spin configuration in {-1, +1}, shape (..., N).

        Returns:
            Logits, shape (..., N). logit_k = log[p(z_k=+1|z_{<k}) / p(z_k=-1|z_{<k})].
        """
        # Map ±1 to (0, 1) for input
        x = (z + 1) / 2  # {0, 1}

        for layer in self.layers[:-1]:
            x = layer(x)
            x = nn.relu(x)
        logits = self.layers[-1](x)
        return logits


def log_prob(model, params, z):
    """Compute log p_θ(z) = Σ_k log p(z_k | z_{<k}).

    Args:
        model: MADE instance
        params: model parameters
        z: spin configuration in {-1, +1}, shape (..., N)

    Returns:
        Log probability, shape (...)
    """
    logits = model.apply(params, z)  # (..., N)
    # z_k ∈ {-1, +1}, map to {0, 1}
    targets = (z + 1) / 2  # (..., N)
    # log p(z_k | z_{<k}) = targets * log σ(logit) + (1-targets) * log(1 - σ(logit))
    #                      = -softplus(-logits) * targets + -softplus(logits) * (1-targets)
    # Numerically stable: log σ(x) = -softplus(-x), log(1-σ(x)) = -softplus(x)
    log_probs_per_site = -jax.nn.softplus(-logits) * targets + (-jax.nn.softplus(logits)) * (1 - targets)
    return jnp.sum(log_probs_per_site, axis=-1)


def sample(model, params, key, num_samples):
    """Autoregressive sampling from the MADE distribution.

    Args:
        model: MADE instance
        params: model parameters
        key: JAX PRNGKey
        num_samples: number of samples to draw

    Returns:
        (samples, log_probs): samples shape (num_samples, N), log_probs shape (num_samples,)
    """
    N = model.n_sites

    def scan_fn(carry, k):
        z, lp = carry
        key_k = jax.random.fold_in(key, k)
        logits = model.apply(params, z)  # (num_samples, N)
        logit_k = logits[:, k]  # (num_samples,)
        # Sample z_k ~ Bernoulli(σ(logit_k))
        prob_plus = jax.nn.sigmoid(logit_k)
        u = jax.random.uniform(key_k, shape=(num_samples,))
        bit = (u < prob_plus).astype(jnp.float32)  # {0, 1}
        spin = 2 * bit - 1  # {-1, +1}
        z = z.at[:, k].set(spin)
        # Accumulate log prob
        lp_k = -jax.nn.softplus(-logit_k) * bit + (-jax.nn.softplus(logit_k)) * (1 - bit)
        lp = lp + lp_k
        return (z, lp), None

    z_init = jnp.zeros((num_samples, N))
    lp_init = jnp.zeros(num_samples)
    (z_final, lp_final), _ = jax.lax.scan(scan_fn, (z_init, lp_init), jnp.arange(N))
    return z_final, lp_final
