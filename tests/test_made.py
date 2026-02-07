"""Tests for MADE autoregressive network."""

import jax
import jax.numpy as jnp
import pytest
from itertools import product

from dsflow_ising.made import MADE, log_prob, sample


@pytest.fixture
def small_model():
    """2x2 = 4 site model for exact enumeration tests."""
    model = MADE(n_sites=4, hidden_dims=(16,))
    key = jax.random.PRNGKey(42)
    params = model.init(key, jnp.ones(4))
    return model, params


class TestAutoregressiveProperty:
    def test_jacobian_mask(self):
        """∂logit_k/∂z_j = 0 for j >= k (autoregressive constraint)."""
        N = 4
        model = MADE(n_sites=N, hidden_dims=(16,))
        key = jax.random.PRNGKey(0)
        z = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(N,))
        params = model.init(key, z)

        def logits_fn(z):
            return model.apply(params, z)

        J = jax.jacobian(logits_fn)(z)  # (N, N)
        # J[k, j] should be 0 for j >= k
        for k in range(N):
            for j in range(k, N):
                assert jnp.isclose(J[k, j], 0.0, atol=1e-6), \
                    f"logit_{k} depends on z_{j}: J[{k},{j}]={J[k,j]}"

    def test_first_output_is_unconditional(self):
        """logit_0 should not depend on any input (it's the prior for z_0)."""
        N = 4
        model = MADE(n_sites=N, hidden_dims=(16,))
        key = jax.random.PRNGKey(0)
        params = model.init(key, jnp.ones(N))

        z1 = jnp.array([1.0, 1.0, 1.0, 1.0])
        z2 = jnp.array([-1.0, -1.0, -1.0, -1.0])
        logits1 = model.apply(params, z1)
        logits2 = model.apply(params, z2)
        assert jnp.isclose(logits1[0], logits2[0], atol=1e-6)


class TestNormalization:
    def test_sum_to_one(self, small_model):
        """For 4-site system, Σ exp(log_prob) over all 16 configs ≈ 1."""
        model, params = small_model
        # Enumerate all 2^4 = 16 configurations
        configs = jnp.array(
            [list(c) for c in product([-1.0, 1.0], repeat=4)]
        )
        lps = jax.vmap(lambda z: log_prob(model, params, z))(configs)
        total = jnp.sum(jnp.exp(lps))
        assert jnp.isclose(total, 1.0, atol=1e-5), f"Sum of probs = {total}"


class TestSamplingConsistency:
    def test_sample_log_probs_match(self, small_model):
        """Log probs from sample() match log_prob() evaluation."""
        model, params = small_model
        key = jax.random.PRNGKey(123)
        samples, sample_lps = sample(model, params, key, num_samples=50)

        # Recompute log probs
        eval_lps = jax.vmap(lambda z: log_prob(model, params, z))(samples)
        assert jnp.allclose(sample_lps, eval_lps, atol=1e-5), \
            f"Max diff: {jnp.max(jnp.abs(sample_lps - eval_lps))}"

    def test_samples_are_spins(self, small_model):
        """All samples should be in {-1, +1}."""
        model, params = small_model
        key = jax.random.PRNGKey(456)
        samples, _ = sample(model, params, key, num_samples=100)
        assert jnp.all((samples == 1.0) | (samples == -1.0))


class TestShapes:
    def test_logits_shape(self):
        N = 16
        model = MADE(n_sites=N, hidden_dims=(32,))
        key = jax.random.PRNGKey(0)
        z = jnp.ones(N)
        params = model.init(key, z)
        logits = model.apply(params, z)
        assert logits.shape == (N,)

    def test_batch_logits(self):
        N = 16
        model = MADE(n_sites=N, hidden_dims=(32,))
        key = jax.random.PRNGKey(0)
        z = jnp.ones((8, N))
        params = model.init(key, jnp.ones(N))
        logits = model.apply(params, z)
        assert logits.shape == (8, N)

    def test_log_prob_shape(self):
        N = 16
        model = MADE(n_sites=N, hidden_dims=(32,))
        key = jax.random.PRNGKey(0)
        params = model.init(key, jnp.ones(N))
        z = jnp.ones((5, N))
        lp = jax.vmap(lambda z: log_prob(model, params, z))(z)
        assert lp.shape == (5,)

    def test_sample_shape(self):
        N = 16
        model = MADE(n_sites=N, hidden_dims=(32,))
        key = jax.random.PRNGKey(0)
        params = model.init(key, jnp.ones(N))
        samples, lps = sample(model, params, key, num_samples=10)
        assert samples.shape == (10, N)
        assert lps.shape == (10,)
