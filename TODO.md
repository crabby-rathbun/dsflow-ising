# TODO

## 1. Exact benchmarks via transfer matrix
- Compute exact free energy F/N, energy E/N, entropy S/N at T_c = 2.269
- Compare variational F_var/N against exact F/N to quantify the gap

## 2. Ablation study: autoregressive only (no flow)
- Train MADE alone (identity flow) as baseline
- Compare F_var, E/N, S/N convergence with and without flow layers
- Sweep flow depth (0, 2, 4, 8 layers) to measure marginal improvement per layer
- Use `layer_free_energy_reduction` diagnostic to show per-layer contribution
