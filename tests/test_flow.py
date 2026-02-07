"""Tests for the full discrete flow (stacked coupling layers)."""

import jax
import jax.numpy as jnp
import pytest
from itertools import product

from dsflow_ising.flow import DiscreteFlow, get_partition


@pytest.fixture
def flow_4x4():
    """4x4 flow with 4 layers."""
    L = 4
    model = DiscreteFlow(L=L, n_layers=4, mask_features=(8, 8))
    key = jax.random.PRNGKey(42)
    z_dummy = jnp.ones(L * L)
    params = model.init(key, z_dummy)
    return model, params, L


class TestBijectivity:
    def test_forward_inverse_roundtrip(self, flow_4x4):
        """flow.inverse(flow(z)) == z for random z."""
        model, params, L = flow_4x4
        key = jax.random.PRNGKey(0)
        z = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(L * L,))

        sigma = model.apply(params, z, use_ste=False)
        z_rec = model.apply(params, sigma, use_ste=False, inverse=True)
        assert jnp.allclose(z_rec, z), f"Max diff: {jnp.max(jnp.abs(z_rec - z))}"

    def test_roundtrip_multiple_seeds(self, flow_4x4):
        """Bijectivity holds across multiple random parameter initializations."""
        L = 4
        for seed in range(5):
            key = jax.random.PRNGKey(seed * 100)
            model = DiscreteFlow(L=L, n_layers=4, mask_features=(8, 8))
            params = model.init(key, jnp.ones(L * L))
            z = jax.random.choice(jax.random.PRNGKey(seed), jnp.array([-1.0, 1.0]), shape=(L * L,))
            sigma = model.apply(params, z, use_ste=False)
            z_rec = model.apply(params, sigma, use_ste=False, inverse=True)
            assert jnp.allclose(z_rec, z), f"Failed for seed={seed}"

    def test_roundtrip_batched(self, flow_4x4):
        """Bijectivity works with batched inputs."""
        model, params, L = flow_4x4
        key = jax.random.PRNGKey(1)
        z = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(20, L * L))
        sigma = model.apply(params, z, use_ste=False)
        z_rec = model.apply(params, sigma, use_ste=False, inverse=True)
        assert jnp.allclose(z_rec, z)

    def test_exhaustive_permutation_2x2(self):
        """For 2x2 system (16 configs), flow output is a permutation of all configs."""
        L = 2
        N = L * L  # 4 sites, 16 configs
        model = DiscreteFlow(L=L, n_layers=4, mask_features=(4, 4))
        key = jax.random.PRNGKey(99)
        params = model.init(key, jnp.ones(N))

        # Enumerate all 2^4 configs
        all_configs = jnp.array(
            [list(c) for c in product([-1.0, 1.0], repeat=N)]
        )  # (16, 4)

        # Apply flow to each
        sigma_all = model.apply(params, all_configs, use_ste=False)

        # Verify all outputs are in {-1, +1}
        assert jnp.all((sigma_all == 1.0) | (sigma_all == -1.0))

        # Verify uniqueness (it's a permutation)
        # Convert to tuples for comparison
        sigma_set = set(tuple(s.tolist()) for s in sigma_all)
        assert len(sigma_set) == 2 ** N, \
            f"Expected {2**N} unique outputs, got {len(sigma_set)}"


class TestComposition:
    def test_alternating_partitions(self):
        """Verify layers alternate even/odd partitions."""
        assert get_partition(0) == "even"
        assert get_partition(1) == "odd"
        assert get_partition(2) == "even"
        assert get_partition(3) == "odd"

    def test_flow_changes_input(self, flow_4x4):
        """Flow should generally change the input (not identity)."""
        model, params, L = flow_4x4
        key = jax.random.PRNGKey(7)
        z = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(L * L,))
        sigma = model.apply(params, z, use_ste=False)
        # It's extremely unlikely that a random flow is the identity
        assert not jnp.allclose(sigma, z), "Flow appears to be identity"

    def test_outputs_are_spins(self, flow_4x4):
        """All flow outputs should be in {-1, +1}."""
        model, params, L = flow_4x4
        key = jax.random.PRNGKey(8)
        z = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(50, L * L))
        sigma = model.apply(params, z, use_ste=False)
        assert jnp.all((sigma == 1.0) | (sigma == -1.0))
