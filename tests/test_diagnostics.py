"""Tests for diagnostics metrics."""

import jax
import jax.numpy as jnp
import pytest

from dsflow_ising.ising import nearest_neighbor_pairs
from dsflow_ising.made import MADE, sample
from dsflow_ising.flow import DiscreteFlow
from dsflow_ising.diagnostics import (
    variational_free_energy, base_entropy, conditional_entropy_profile,
    layer_free_energy_reduction,
)


@pytest.fixture
def diag_setup():
    """2x2 system for diagnostics tests."""
    L = 2
    N = L * L
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)

    made_model = MADE(n_sites=N, hidden_dims=(16,))
    made_params = made_model.init(k1, jnp.ones(N))

    flow_model = DiscreteFlow(L=L, n_layers=2, mask_features=(4, 4))
    flow_params = flow_model.init(k2, jnp.ones(N))

    pairs = nearest_neighbor_pairs(L)
    return made_model, made_params, flow_model, flow_params, pairs, L, N


class TestBaseEntropy:
    def test_entropy_nonnegative(self, diag_setup):
        made_model, made_params, *_ = diag_setup
        key = jax.random.PRNGKey(10)
        H = base_entropy(made_model, made_params, key, num_samples=500)
        assert H >= -0.01, f"Entropy is negative: {H}"

    def test_entropy_bounded_by_uniform(self, diag_setup):
        """Entropy should be at most N * ln(2) (uniform distribution)."""
        made_model, made_params, _, _, _, _, N = diag_setup
        key = jax.random.PRNGKey(11)
        H = base_entropy(made_model, made_params, key, num_samples=1000)
        max_entropy = N * jnp.log(2.0)
        assert H <= max_entropy + 0.1, f"Entropy {H} exceeds max {max_entropy}"


class TestConditionalEntropyProfile:
    def test_profile_shape(self, diag_setup):
        made_model, made_params, *_ = diag_setup
        N = diag_setup[-1]
        key = jax.random.PRNGKey(12)
        profile = conditional_entropy_profile(made_model, made_params, key, num_samples=500)
        assert profile.shape == (N,)

    def test_per_site_nonnegative(self, diag_setup):
        made_model, made_params, *_ = diag_setup
        key = jax.random.PRNGKey(13)
        profile = conditional_entropy_profile(made_model, made_params, key, num_samples=500)
        assert jnp.all(profile >= -0.01)

    def test_sum_approximates_total_entropy(self, diag_setup):
        """Sum of conditional entropies should approximate total entropy."""
        made_model, made_params, *_ = diag_setup
        key = jax.random.PRNGKey(14)
        k1, k2 = jax.random.split(key)
        profile = conditional_entropy_profile(made_model, made_params, k1, num_samples=2000)
        H = base_entropy(made_model, made_params, k2, num_samples=2000)
        # H[p] = Σ_k H(z_k | z_{<k}) — should be approximately equal
        assert jnp.isclose(jnp.sum(profile), H, atol=0.3), \
            f"Sum of conditionals {jnp.sum(profile):.3f} != total entropy {H:.3f}"


class TestVariationalFreeEnergy:
    def test_is_scalar(self, diag_setup):
        made_model, made_params, flow_model, flow_params, pairs, L, N = diag_setup
        key = jax.random.PRNGKey(20)
        F = variational_free_energy(
            made_model, made_params, flow_model, flow_params,
            pairs, 1.0, 2.0, key, num_samples=100,
        )
        assert F.shape == ()


class TestLayerFreeEnergyReduction:
    def test_deltas_shape(self, diag_setup):
        made_model, made_params, flow_model, flow_params, pairs, L, N = diag_setup
        key = jax.random.PRNGKey(30)
        deltas = layer_free_energy_reduction(
            made_model, made_params, flow_model, flow_params,
            pairs, 1.0, 2.0, key, num_samples=200,
        )
        assert deltas.shape == (flow_model.n_layers,)

    def test_deltas_sum_to_total_improvement(self, diag_setup):
        """Sum of per-layer ΔF should equal total F_var improvement (0 layers vs all)."""
        made_model, made_params, flow_model, flow_params, pairs, L, N = diag_setup
        T, J = 2.0, 1.0
        key = jax.random.PRNGKey(31)
        k1, k2, k3 = jax.random.split(key, 3)

        deltas = layer_free_energy_reduction(
            made_model, made_params, flow_model, flow_params,
            pairs, J, T, k1, num_samples=500,
        )

        # F_var with no flow (identity)
        z_samples, z_log_probs = sample(made_model, made_params, k2, num_samples=500)
        from dsflow_ising.ising import energy as compute_energy
        energies_no_flow = jax.vmap(lambda s: compute_energy(s, pairs, J))(z_samples)
        f_no_flow = jnp.mean(energies_no_flow + T * z_log_probs)

        # F_var with full flow
        f_full = variational_free_energy(
            made_model, made_params, flow_model, flow_params,
            pairs, J, T, k3, num_samples=500,
        )

        total_from_deltas = jnp.sum(deltas)
        total_actual = f_no_flow - f_full
        # These use different samples so allow reasonable tolerance
        assert jnp.isclose(total_from_deltas, total_actual, atol=1.0), \
            f"Sum of deltas={total_from_deltas:.3f} != actual improvement={total_actual:.3f}"
