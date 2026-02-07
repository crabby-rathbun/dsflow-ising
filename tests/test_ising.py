"""Tests for Ising model energy and lattice utilities."""

import jax.numpy as jnp
import pytest

from dsflow_ising.ising import energy, magnetization, nearest_neighbor_pairs


class TestNearestNeighborPairs:
    def test_pair_count(self):
        """Periodic square lattice has 2*L*L bonds."""
        for L in [2, 4, 8, 16]:
            pairs = nearest_neighbor_pairs(L)
            assert pairs.shape == (2 * L * L, 2)

    def test_indices_in_range(self):
        L = 4
        pairs = nearest_neighbor_pairs(L)
        assert jnp.all(pairs >= 0)
        assert jnp.all(pairs < L * L)

    def test_no_self_loops(self):
        L = 4
        pairs = nearest_neighbor_pairs(L)
        assert jnp.all(pairs[:, 0] != pairs[:, 1])


class TestEnergy:
    def test_all_up(self):
        """All spins +1: E = -J * 2N (each of 2N bonds contributes -J)."""
        L = 4
        N = L * L
        pairs = nearest_neighbor_pairs(L)
        sigma = jnp.ones(N)
        E = energy(sigma, pairs, J=1.0)
        assert jnp.isclose(E, -2 * N)

    def test_all_down(self):
        """All spins -1: same energy as all +1."""
        L = 4
        N = L * L
        pairs = nearest_neighbor_pairs(L)
        sigma = -jnp.ones(N)
        E = energy(sigma, pairs, J=1.0)
        assert jnp.isclose(E, -2 * N)

    def test_checkerboard(self):
        """Checkerboard: every bond is antiferromagnetic, E = +J * 2N = +2N."""
        L = 4
        N = L * L
        pairs = nearest_neighbor_pairs(L)
        # Checkerboard: sigma_i = (-1)^(row+col)
        sigma = jnp.array([(-1) ** (i // L + i % L) for i in range(N)], dtype=jnp.float32)
        E = energy(sigma, pairs, J=1.0)
        assert jnp.isclose(E, 2 * N)

    def test_single_flip_energy_change(self):
        """Flipping one spin in all-up config changes energy by +2J * (num_neighbors)."""
        L = 4
        N = L * L
        pairs = nearest_neighbor_pairs(L)
        J = 1.0
        sigma_up = jnp.ones(N)
        E_up = energy(sigma_up, pairs, J=J)

        # Flip spin at site (1,1) = index 5 — has 4 neighbors on periodic lattice
        sigma_flip = sigma_up.at[5].set(-1.0)
        E_flip = energy(sigma_flip, pairs, J=J)
        # Each of the 4 bonds to flipped spin changes from -J to +J: ΔE = 4 * 2J = 8J
        assert jnp.isclose(E_flip - E_up, 8 * J)

    def test_batch_dimension(self):
        """Energy works with batched inputs."""
        L = 4
        N = L * L
        pairs = nearest_neighbor_pairs(L)
        batch = 10
        sigma = jnp.ones((batch, N))
        E = energy(sigma, pairs)
        assert E.shape == (batch,)
        assert jnp.allclose(E, -2 * N)

    def test_coupling_constant(self):
        """Energy scales with J."""
        L = 4
        pairs = nearest_neighbor_pairs(L)
        sigma = jnp.ones(L * L)
        E1 = energy(sigma, pairs, J=1.0)
        E2 = energy(sigma, pairs, J=2.0)
        assert jnp.isclose(E2, 2 * E1)


class TestMagnetization:
    def test_all_up(self):
        sigma = jnp.ones(16)
        assert jnp.isclose(magnetization(sigma), 1.0)

    def test_all_down(self):
        sigma = -jnp.ones(16)
        assert jnp.isclose(magnetization(sigma), 1.0)

    def test_half_and_half(self):
        sigma = jnp.array([1, 1, -1, -1], dtype=jnp.float32)
        assert jnp.isclose(magnetization(sigma), 0.0)

    def test_batch(self):
        sigma = jnp.ones((5, 16))
        m = magnetization(sigma)
        assert m.shape == (5,)
