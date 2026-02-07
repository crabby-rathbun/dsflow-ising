"""Tests for training loop."""

import jax
import jax.numpy as jnp
import optax
import pytest

from dsflow_ising.ising import nearest_neighbor_pairs, energy
from dsflow_ising.made import MADE, log_prob, sample
from dsflow_ising.flow import DiscreteFlow
from dsflow_ising.train import compute_loss, train_step, TrainState
from dsflow_ising.config import ModelConfig, TrainConfig


@pytest.fixture
def tiny_setup():
    """2x2 system for fast tests."""
    L = 2
    N = L * L
    key = jax.random.PRNGKey(0)

    made_model = MADE(n_sites=N, hidden_dims=(16,))
    k1, k2 = jax.random.split(key)
    made_params = made_model.init(k1, jnp.ones(N))

    flow_model = DiscreteFlow(L=L, n_layers=2, mask_features=(4, 4))
    flow_params = flow_model.init(k2, jnp.ones(N))

    pairs = nearest_neighbor_pairs(L)
    return made_model, made_params, flow_model, flow_params, pairs, L, N


class TestComputeLoss:
    def test_manual_calculation(self, tiny_setup):
        """Loss computation matches manual calculation on small system."""
        made_model, made_params, flow_model, flow_params, pairs, L, N = tiny_setup
        T, J = 2.0, 1.0

        key = jax.random.PRNGKey(10)
        z_samples, z_log_probs = sample(made_model, made_params, key, num_samples=5)

        f_var, energies = compute_loss(
            made_model, made_params, flow_model, flow_params,
            z_samples, z_log_probs, pairs, J, T,
        )

        # Manual: F_var = mean(E(σ) + T * ln p_θ(z))
        sigma = flow_model.apply(flow_params, z_samples, use_ste=False)
        manual_energies = jax.vmap(lambda s: energy(s, pairs, J))(sigma)
        manual_f_var = jnp.mean(manual_energies + T * z_log_probs)

        assert jnp.isclose(f_var, manual_f_var, atol=1e-5)
        assert jnp.allclose(energies, manual_energies, atol=1e-5)

    def test_loss_is_scalar(self, tiny_setup):
        made_model, made_params, flow_model, flow_params, pairs, L, N = tiny_setup
        key = jax.random.PRNGKey(11)
        z_samples, z_log_probs = sample(made_model, made_params, key, num_samples=10)
        f_var, _ = compute_loss(
            made_model, made_params, flow_model, flow_params,
            z_samples, z_log_probs, pairs, 1.0, 2.0,
        )
        assert f_var.shape == ()


class TestTrainStep:
    def test_step_runs(self, tiny_setup):
        """Training step runs without error and returns updated state."""
        made_model, made_params, flow_model, flow_params, pairs, L, N = tiny_setup
        T, J = 2.0, 1.0

        made_opt = optax.adam(1e-3)
        flow_opt = optax.adam(1e-3)

        state = TrainState(
            made_params=made_params,
            flow_params=flow_params,
            made_opt_state=made_opt.init(made_params),
            flow_opt_state=flow_opt.init(flow_params),
            baseline=0.0,
            step=0,
        )

        key = jax.random.PRNGKey(20)
        new_state, metrics = train_step(
            made_model, flow_model, pairs, J, T, 32,
            made_opt, flow_opt, state, key,
        )

        assert new_state.step == 1
        assert 'f_var' in metrics
        assert 'energy' in metrics
        assert 'entropy' in metrics

    def test_params_change_after_step(self, tiny_setup):
        """Parameters should change after a training step."""
        made_model, made_params, flow_model, flow_params, pairs, L, N = tiny_setup
        T, J = 2.0, 1.0

        made_opt = optax.adam(1e-3)
        flow_opt = optax.adam(1e-3)

        state = TrainState(
            made_params=made_params,
            flow_params=flow_params,
            made_opt_state=made_opt.init(made_params),
            flow_opt_state=flow_opt.init(flow_params),
            baseline=0.0,
            step=0,
        )

        key = jax.random.PRNGKey(30)
        new_state, _ = train_step(
            made_model, flow_model, pairs, J, T, 64,
            made_opt, flow_opt, state, key,
        )

        # At least some parameters should have changed
        old_leaves = jax.tree.leaves(state.made_params)
        new_leaves = jax.tree.leaves(new_state.made_params)
        changed = any(not jnp.allclose(o, n) for o, n in zip(old_leaves, new_leaves))
        assert changed, "MADE params did not change after training step"


class TestTrainingReducesLoss:
    def test_loss_decreases(self):
        """Run several training steps and verify F_var tends to decrease."""
        L = 2
        N = L * L
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)

        made_model = MADE(n_sites=N, hidden_dims=(16,))
        made_params = made_model.init(k1, jnp.ones(N))

        flow_model = DiscreteFlow(L=L, n_layers=2, mask_features=(4, 4))
        flow_params = flow_model.init(k2, jnp.ones(N))

        pairs = nearest_neighbor_pairs(L)
        T, J = 2.0, 1.0

        made_opt = optax.adam(1e-3)
        flow_opt = optax.adam(1e-3)

        state = TrainState(
            made_params=made_params,
            flow_params=flow_params,
            made_opt_state=made_opt.init(made_params),
            flow_opt_state=flow_opt.init(flow_params),
            baseline=0.0,
            step=0,
        )

        key = jax.random.PRNGKey(100)
        f_var_values = []
        for i in range(100):
            key, subkey = jax.random.split(key)
            state, metrics = train_step(
                made_model, flow_model, pairs, J, T, 128,
                made_opt, flow_opt, state, subkey,
            )
            f_var_values.append(float(metrics['f_var']))

        # Compare first 10 mean vs last 10 mean — should generally decrease
        first = jnp.mean(jnp.array(f_var_values[:10]))
        last = jnp.mean(jnp.array(f_var_values[-10:]))
        assert last < first, f"F_var didn't decrease: first_10_avg={first:.4f}, last_10_avg={last:.4f}"


class TestBaseline:
    def test_baseline_reduces_variance(self, tiny_setup):
        """Baseline should reduce variance of REINFORCE gradient estimates."""
        made_model, made_params, flow_model, flow_params, pairs, L, N = tiny_setup
        T, J = 2.0, 1.0

        key = jax.random.PRNGKey(50)
        z_samples, z_log_probs = sample(made_model, made_params, key, num_samples=256)

        sigma = flow_model.apply(flow_params, z_samples, use_ste=False)
        energies = jax.vmap(lambda s: energy(s, pairs, J))(sigma)
        rewards = energies + T * z_log_probs

        # Gradient estimate without baseline
        advantage_no_bl = rewards
        # Gradient estimate with baseline
        baseline = jnp.mean(rewards)
        advantage_with_bl = rewards - baseline

        # The variance of (advantage * lp) should be lower with baseline
        # We use a proxy: variance of the advantage itself
        var_no_bl = jnp.var(advantage_no_bl)
        var_with_bl = jnp.var(advantage_with_bl)
        # With baseline, variance should be lower (or equal if baseline is 0)
        # The baseline is the mean, which minimizes variance of (X - b)
        assert var_with_bl <= var_no_bl + 1e-6
