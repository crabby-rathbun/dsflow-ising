"""Exact partition function for the finite-size 2D Ising model on a torus.

Implements the Kaufman formula (Eq. 39 from Kaufman 1949) for an L×L periodic
square lattice. Thermodynamic quantities are obtained via numerical differentiation.
"""

import numpy as np


def _logsumexp(a):
    """Numerically stable log-sum-exp."""
    a = np.asarray(a, dtype=np.float64)
    a_max = np.max(a)
    if not np.isfinite(a_max):
        return a_max
    return a_max + np.log(np.sum(np.exp(a - a_max)))


def _signed_logsumexp(log_abs, signs):
    """Compute log(sum(signs * exp(log_abs))) where signs are ±1."""
    log_abs = np.asarray(log_abs, dtype=np.float64)
    signs = np.asarray(signs, dtype=np.float64)
    a_max = np.max(log_abs)
    if not np.isfinite(a_max):
        return a_max
    val = np.sum(signs * np.exp(log_abs - a_max))
    return a_max + np.log(val)


def log_partition(L, T, J=1.0):
    """Exact log partition function for L×L periodic 2D Ising model.

    Uses the Kaufman formula with m = n = L:
        Z = (1/2)(2 sinh 2H)^{N/2} [T1 + T2 + T3 + T4]
    where the four terms are products over cosh/sinh of gamma values.

    Args:
        L: lattice side length
        T: temperature (k_B = 1)
        J: coupling constant (> 0 for ferromagnetic)

    Returns:
        ln Z (float64)
    """
    N = L * L
    H = J / T
    Hstar = np.arctanh(np.exp(-2.0 * H))

    # gamma_r for r = 1, ..., 2L
    # cosh(gamma_r) = cosh(2H*) cosh(2H) - sinh(2H*) sinh(2H) cos(r*pi/L)
    r_vals = np.arange(1, 2 * L + 1, dtype=np.float64)
    cos_vals = np.cos(r_vals * np.pi / L)
    cosh_gamma = (np.cosh(2.0 * Hstar) * np.cosh(2.0 * H)
                  - np.sinh(2.0 * Hstar) * np.sinh(2.0 * H) * cos_vals)
    cosh_gamma = np.maximum(cosh_gamma, 1.0)
    gamma = np.arccosh(cosh_gamma)

    # Sign fix: for r=2L (cos=1), gamma_{2L} = 2H* - 2H analytically.
    # arccosh always returns >= 0, but the physical value can be negative
    # (below T_c where H > H*). This affects the sign of the T2 product.
    gamma[2 * L - 1] = 2.0 * H - 2.0 * Hstar  # signed: positive below Tc, negative above

    # gamma_{2r} for r=1..L: 0-based indices 1, 3, ..., 2L-1
    gamma_even = gamma[1::2]
    # gamma_{2r-1} for r=1..L: 0-based indices 0, 2, ..., 2L-2
    gamma_odd = gamma[0::2]

    half_L = L / 2.0

    # T1 = prod_r 2*cosh(L/2 * gamma_{2r})  (cosh is even, sign doesn't matter)
    log_T1 = np.sum(np.log(2.0 * np.cosh(half_L * gamma_even)))

    # T2 = prod_r 2*sinh(L/2 * gamma_{2r})
    # sinh preserves sign, so T2 can be negative (below T_c)
    s_even = np.sinh(half_L * gamma_even)
    T2_sign = np.prod(np.sign(s_even))
    with np.errstate(divide='ignore'):
        log_abs_T2 = np.sum(np.where(
            np.abs(s_even) > 0, np.log(2.0 * np.abs(s_even)), -np.inf))

    # T3 = prod_r 2*cosh(L/2 * gamma_{2r-1})
    log_T3 = np.sum(np.log(2.0 * np.cosh(half_L * gamma_odd)))

    # T4 = prod_r 2*sinh(L/2 * gamma_{2r-1})
    # gamma_odd are always > 0, so T4 > 0
    with np.errstate(divide='ignore'):
        s_odd = np.sinh(half_L * gamma_odd)
        log_T4 = np.sum(np.where(s_odd > 0, np.log(2.0 * s_odd), -np.inf))

    # Combine: bracket = T1 + T2 + T3 + T4 (T2 may be negative)
    # Use logsumexp with signed terms
    log_terms = np.array([log_T1, log_abs_T2, log_T3, log_T4])
    signs = np.array([1.0, T2_sign, 1.0, 1.0])
    log_bracket = _signed_logsumexp(log_terms, signs)
    log_Z = -np.log(2.0) + (N / 2.0) * np.log(2.0 * np.sinh(2.0 * H)) + log_bracket

    return log_Z


def exact_free_energy_per_site(L, T, J=1.0):
    """Exact free energy per site: f = -T ln Z / N."""
    N = L * L
    return -T * log_partition(L, T, J) / N


def exact_energy_per_site(L, T, J=1.0, dT=1e-6):
    """Exact energy per site via numerical differentiation.

    E = -d(ln Z)/d(beta), computed by central finite difference on beta.
    """
    N = L * L
    beta = 1.0 / T
    lnZ_plus = log_partition(L, 1.0 / (beta + dT), J)
    lnZ_minus = log_partition(L, 1.0 / (beta - dT), J)
    E = -(lnZ_plus - lnZ_minus) / (2.0 * dT)
    return E / N


def exact_entropy_per_site(L, T, J=1.0):
    """Exact entropy per site: s = (e - f) / T."""
    e = exact_energy_per_site(L, T, J)
    f = exact_free_energy_per_site(L, T, J)
    return (e - f) / T
