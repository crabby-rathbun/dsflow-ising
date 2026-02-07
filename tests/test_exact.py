"""Tests for exact partition function (Kaufman formula)."""

import numpy as np
import pytest
from itertools import product

from dsflow_ising.exact import (
    log_partition, exact_free_energy_per_site,
    exact_energy_per_site, exact_entropy_per_site,
)
from dsflow_ising.ising import nearest_neighbor_pairs


def brute_force_log_Z(L, T, J=1.0):
    """Compute ln Z by enumerating all 2^N configurations."""
    N = L * L
    pairs = nearest_neighbor_pairs(L)
    # Convert JAX array to numpy
    pairs_np = np.array(pairs)

    beta = 1.0 / T
    log_weights = []
    for bits in product([0, 1], repeat=N):
        sigma = np.array([2 * b - 1 for b in bits], dtype=np.float64)
        E = -J * np.sum(sigma[pairs_np[:, 0]] * sigma[pairs_np[:, 1]])
        log_weights.append(-beta * E)

    log_weights = np.array(log_weights)
    max_lw = np.max(log_weights)
    return max_lw + np.log(np.sum(np.exp(log_weights - max_lw)))


class TestLogPartitionVsBruteForce:
    @pytest.mark.parametrize("L", [2, 4])
    @pytest.mark.parametrize("T", [1.0, 2.269, 4.0])
    def test_matches_brute_force(self, L, T):
        """Kaufman formula matches brute-force enumeration."""
        lnZ_exact = log_partition(L, T)
        lnZ_brute = brute_force_log_Z(L, T)
        assert np.isclose(lnZ_exact, lnZ_brute, rtol=1e-10), \
            f"L={L}, T={T}: exact={lnZ_exact:.10f}, brute={lnZ_brute:.10f}"


class TestThermodynamicConsistency:
    def test_free_energy_below_zero(self):
        """Free energy per site should be negative."""
        for L in [4, 8, 16]:
            f = exact_free_energy_per_site(L, 2.269)
            assert f < 0, f"L={L}: f={f}"

    def test_energy_between_bounds(self):
        """Energy per site should be between -2J and 0 for T > 0."""
        for L in [4, 8, 16]:
            e = exact_energy_per_site(L, 2.269)
            assert -2.0 <= e <= 0.0, f"L={L}: e={e}"

    def test_entropy_nonneg(self):
        """Entropy per site should be non-negative."""
        for L in [4, 8, 16]:
            s = exact_entropy_per_site(L, 2.269)
            assert s >= -1e-10, f"L={L}: s={s}"

    def test_thermodynamic_identity(self):
        """f = e - T*s."""
        L, T = 8, 2.269
        f = exact_free_energy_per_site(L, T)
        e = exact_energy_per_site(L, T)
        s = exact_entropy_per_site(L, T)
        assert np.isclose(f, e - T * s, rtol=1e-6), \
            f"f={f:.6f}, e - Ts = {e - T * s:.6f}"


class TestLimits:
    def test_high_T_entropy(self):
        """At high T, entropy per site → ln 2 (uniform distribution)."""
        s = exact_entropy_per_site(8, T=100.0)
        assert np.isclose(s, np.log(2), rtol=0.01), f"s={s}, ln2={np.log(2)}"

    def test_low_T_energy(self):
        """At low T, energy per site → -2J (ground state)."""
        e = exact_energy_per_site(8, T=0.5)
        assert np.isclose(e, -2.0, atol=0.01), f"e={e}"

    def test_size_convergence(self):
        """Larger L should converge toward thermodynamic limit."""
        T = 2.269
        f_vals = [exact_free_energy_per_site(L, T) for L in [4, 8, 16, 32]]
        # Differences should shrink
        diffs = [abs(f_vals[i+1] - f_vals[i]) for i in range(len(f_vals) - 1)]
        assert diffs[1] < diffs[0], f"Not converging: diffs={diffs}"
