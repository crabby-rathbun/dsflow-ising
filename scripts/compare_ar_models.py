#!/usr/bin/env python
"""Compare two autoregressive-only runs against exact results."""

import csv
import numpy as np
import sys
import matplotlib.pyplot as plt

from dsflow_ising.exact import (
    exact_free_energy_per_site,
    exact_energy_per_site,
    exact_entropy_per_site,
)

L = 8
N = L * L
T = 2.269
BATCH_SIZE = 256
N_BINS = 100

f_exact = exact_free_energy_per_site(L, T)
e_exact = exact_energy_per_site(L, T)
s_exact = exact_entropy_per_site(L, T)

print(f"L={L}, T={T}, N={N}")
print(f"\nExact: F/N={f_exact:.6f}, E/N={e_exact:.6f}, S/N={s_exact:.6f}")


def read_csv(path):
    steps, f_var, energy, entropy = [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row['step']))
            f_var.append(float(row['f_var']))
            energy.append(float(row['energy']))
            entropy.append(float(row['entropy']))
    return np.array(steps), np.array(f_var), np.array(energy), np.array(entropy)


def bin_stats(steps, vals, n_bins):
    """Bin data into n_bins chunks, return bin centers, means, and SEM."""
    n = len(steps)
    chunk = max(1, n // n_bins)
    centers, means, sems = [], [], []
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        centers.append(np.mean(steps[sl]))
        means.append(np.mean(vals[sl]))
        sems.append(np.std(vals[sl]) / np.sqrt(len(vals[sl])))
    return np.array(centers), np.array(means), np.array(sems)


# --- Data ---
runs = [
    ("AR small (h=256, 33K)",  "logs/ar_small.csv", 33088,  'tab:blue'),
    ("AR large (h=1024, 132K)", "logs/ar_large.csv", 132160, 'tab:red'),
]

curves = []
for label, path, n_params, color in runs:
    try:
        steps, f_var, energy, entropy = read_csv(path)
    except FileNotFoundError:
        print(f"{label}: not found at {path}")
        continue
    curves.append((label, steps, f_var, energy, entropy, n_params, color))

    tail = min(200, len(steps))
    f_avg = np.mean(f_var[-tail:]) / N
    e_avg = np.mean(energy[-tail:]) / N
    print(f"  {label}: F/N={f_avg:.6f} (dF={f_avg - f_exact:+.6f}), "
          f"E/N={e_avg:.6f}, {len(steps)} steps")

if not curves:
    print("No CSV files found.")
    sys.exit(1)

# --- Plot ---
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
ylabels = ['F/N', 'E/N', 'S/N']
exact_vals = [f_exact, e_exact, s_exact]

for label, steps, f_var, energy, entropy, n_params, color in curves:
    for ax, vals in zip(axes, [f_var / N, energy / N, entropy / N]):
        c, m, s = bin_stats(steps, vals, N_BINS)
        ax.plot(c, m, color=color, lw=2, label=label)
        ax.fill_between(c, m - s, m + s, color=color, alpha=0.2)

for ax, val, ylabel in zip(axes, exact_vals, ylabels):
    ax.axhline(val, color='black', ls='--', lw=1.5, label=f'Exact = {val:.4f}')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Training Step', fontsize=12)
fig.suptitle(f'Autoregressive-only MADE: L={L}, T={T}', fontsize=14)
fig.tight_layout()
fig.savefig('logs/compare_ar_models.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to logs/compare_ar_models.png")
plt.show()
