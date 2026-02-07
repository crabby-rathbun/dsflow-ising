"""Tests for discrete coupling layers."""

import jax
import jax.numpy as jnp
import pytest

from dsflow_ising.coupling import (
    MaskNet, checkerboard_indices, forward_layer, inverse_layer, _ste_sign,
)


@pytest.fixture
def layer_4x4():
    """4x4 coupling layer with random params."""
    L = 4
    mask_net = MaskNet(L=L, features=(8, 8))
    key = jax.random.PRNGKey(42)
    dummy_grid = jnp.zeros((L, L))
    params = mask_net.init(key, dummy_grid)
    return mask_net, params, L


class TestCheckerboard:
    def test_partition_sizes(self):
        L = 4
        a, b = checkerboard_indices(L, "even")
        assert len(a) == L * L // 2
        assert len(b) == L * L // 2

    def test_partitions_cover_all_sites(self):
        L = 4
        a, b = checkerboard_indices(L, "even")
        all_sites = jnp.sort(jnp.concatenate([a, b]))
        assert jnp.array_equal(all_sites, jnp.arange(L * L))

    def test_even_odd_swap(self):
        L = 4
        a_even, b_even = checkerboard_indices(L, "even")
        a_odd, b_odd = checkerboard_indices(L, "odd")
        assert jnp.array_equal(jnp.sort(a_even), jnp.sort(b_odd))
        assert jnp.array_equal(jnp.sort(b_even), jnp.sort(a_odd))


class TestSelfInverse:
    def test_inverse_recovers_input(self, layer_4x4):
        """forward(forward(z)) == z (self-inverse property)."""
        mask_net, params, L = layer_4x4
        key = jax.random.PRNGKey(0)
        z = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(L * L,))
        sigma = forward_layer(mask_net, params, z, L, "even", use_ste=False)
        z_recovered = inverse_layer(mask_net, params, sigma, L, "even", use_ste=False)
        assert jnp.allclose(z_recovered, z), f"Max diff: {jnp.max(jnp.abs(z_recovered - z))}"

    def test_inverse_recovers_batch(self, layer_4x4):
        """Self-inverse works with batch dimension."""
        mask_net, params, L = layer_4x4
        key = jax.random.PRNGKey(1)
        z = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(10, L * L))
        sigma = forward_layer(mask_net, params, z, L, "odd", use_ste=False)
        z_recovered = inverse_layer(mask_net, params, sigma, L, "odd", use_ste=False)
        assert jnp.allclose(z_recovered, z)

    def test_inverse_multiple_partitions(self, layer_4x4):
        """Self-inverse holds for both even and odd partitions."""
        mask_net, params, L = layer_4x4
        key = jax.random.PRNGKey(2)
        z = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(L * L,))
        for part in ["even", "odd"]:
            sigma = forward_layer(mask_net, params, z, L, part, use_ste=False)
            z_rec = inverse_layer(mask_net, params, sigma, L, part, use_ste=False)
            assert jnp.allclose(z_rec, z), f"Failed for partition={part}"


class TestPassthrough:
    def test_a_sublattice_unchanged(self, layer_4x4):
        """A-sublattice spins pass through unchanged."""
        mask_net, params, L = layer_4x4
        key = jax.random.PRNGKey(3)
        z = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(L * L,))
        a_idx, _ = checkerboard_indices(L, "even")
        sigma = forward_layer(mask_net, params, z, L, "even", use_ste=False)
        assert jnp.allclose(sigma[a_idx], z[a_idx])


class TestOutputValues:
    def test_outputs_are_spins(self, layer_4x4):
        """All outputs should be exactly ±1."""
        mask_net, params, L = layer_4x4
        key = jax.random.PRNGKey(4)
        z = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(20, L * L))
        sigma = forward_layer(mask_net, params, z, L, "even", use_ste=False)
        assert jnp.all((sigma == 1.0) | (sigma == -1.0))


class TestDifferentSizes:
    @pytest.mark.parametrize("L", [2, 4, 8])
    def test_works_for_different_L(self, L):
        mask_net = MaskNet(L=L, features=(8,))
        key = jax.random.PRNGKey(5)
        dummy = jnp.zeros((L, L))
        params = mask_net.init(key, dummy)
        z = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(L * L,))
        sigma = forward_layer(mask_net, params, z, L, "even", use_ste=False)
        assert sigma.shape == (L * L,)
        z_rec = inverse_layer(mask_net, params, sigma, L, "even", use_ste=False)
        assert jnp.allclose(z_rec, z)


class TestSTE:
    def test_gradient_flows(self, layer_4x4):
        """With STE, gradients w.r.t. mask_net params should be nonzero."""
        mask_net, params, L = layer_4x4
        key = jax.random.PRNGKey(6)
        z = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(L * L,))

        def loss_fn(p):
            sigma = forward_layer(mask_net, p, z, L, "even", use_ste=True)
            return jnp.sum(sigma)

        grads = jax.grad(loss_fn)(params)
        # At least some gradients should be nonzero
        grad_leaves = jax.tree.leaves(grads)
        has_nonzero = any(jnp.any(g != 0) for g in grad_leaves)
        assert has_nonzero, "All gradients are zero — STE not working"
