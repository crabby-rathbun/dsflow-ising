#!/usr/bin/env python
"""Plot training metrics from CSV log files."""

import argparse
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    steps, f_var, energy, entropy = [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['step']))
            f_var.append(float(row['f_var']))
            energy.append(float(row['energy']))
            entropy.append(float(row['entropy']))
    return np.array(steps), np.array(f_var), np.array(energy), np.array(entropy)


def smooth(x, window):
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='valid')


def main():
    parser = argparse.ArgumentParser(description="Plot training log")
    parser.add_argument("log_files", nargs='+', help="CSV log files (label:path or just path)")
    parser.add_argument("-o", "--output", default="train_plot.png", help="Output image file")
    parser.add_argument("--L", type=int, required=True, help="Lattice side length")
    parser.add_argument("--T", type=float, default=2.269, help="Temperature (for exact lines)")
    args = parser.parse_args()

    N = args.L ** 2
    colors = ['b', 'r', 'g', 'm', 'c']

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    for i, entry in enumerate(args.log_files):
        if ':' in entry:
            label, path = entry.split(':', 1)
        else:
            label = entry.replace('.csv', '').replace('train_', '')
            path = entry

        c = colors[i % len(colors)]
        steps, f_var, energy, entropy = read_csv(path)
        f_n, e_n, s_n = f_var / N, energy / N, entropy / N

        window = max(1, len(steps) // 50)
        for ax, data, ylabel in zip(axes, [f_n, e_n, s_n], ['F/N', 'E/N', 'S/N']):
            ax.plot(steps, data, color=c, alpha=0.15)
            ax.plot(steps[window-1:], smooth(data, window), color=c, lw=2, label=label)

    # Exact reference lines
    from dsflow_ising.exact import (
        exact_free_energy_per_site, exact_energy_per_site, exact_entropy_per_site,
    )
    f_exact = exact_free_energy_per_site(args.L, args.T)
    e_exact = exact_energy_per_site(args.L, args.T)
    s_exact = exact_entropy_per_site(args.L, args.T)

    for ax, val in zip(axes, [f_exact, e_exact, s_exact]):
        ax.axhline(val, color='k', ls='--', lw=1.5, label=f'exact {val:.4f}')

    axes[0].set_ylabel('F/N')
    axes[0].set_title('Variational Free Energy per Site')
    axes[1].set_ylabel('E/N')
    axes[1].set_title('Energy per Site ⟨E(σ)⟩/N')
    axes[2].set_ylabel('S/N')
    axes[2].set_title('Entropy per Site H[p_θ]/N')
    axes[2].set_xlabel('Training Step')

    for ax in axes:
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
