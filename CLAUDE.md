# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project: **Discrete Normalizing Flows as Variational Ansatze for Classical Statistical Mechanics**. Combines discrete normalizing flows with autoregressive variational models to study the 2D Ising model while preserving exact, tractable log-probabilities and rigorous variational free-energy bounds.

The `doc/` directory contains the LaTeX paper. The `dsflow_ising/` package implements the full training pipeline.

## Build & Test Commands

```bash
pip install -e ".[test]"       # Install in dev mode
pytest tests/ -v               # Run all 55 tests
pytest tests/test_ising.py -v  # Run a single module's tests
python scripts/run_training.py --L 4 --num-steps 200  # Smoke test
```

## Package Structure

```
dsflow_ising/
├── ising.py        # Ising energy, nearest-neighbor pairs, magnetization
├── made.py         # MADE autoregressive network (Flax linen)
├── coupling.py     # Discrete coupling layer (checkerboard partition, STE)
├── flow.py         # DiscreteFlow: stacked coupling layers
├── train.py        # Training loop (REINFORCE for θ, STE for φ)
├── diagnostics.py  # Free energy, entropy, per-layer metrics
└── config.py       # ModelConfig, TrainConfig dataclasses

tests/              # One test file per module (55 tests total)
scripts/
└── run_training.py # CLI entry point with argparse
```

## Core Theoretical Framework

1. **Base distribution** (`p_θ(z)`): MADE autoregressive model over latent binary spins `z ∈ {±1}^N`
2. **Discrete flow** (`f_φ`): Bijective map `{±1}^N → {±1}^N` via coupling layers with checkerboard-partition ConvNet mask networks
3. **Physical distribution**: `q(σ) = p_θ(f_φ⁻¹(σ))` — exact log-prob via change of variables (no Jacobian needed for discrete bijections)

## Key Implementation Details

- **Flax Linen API** (`nn.Module` / `nn.compact`) with JAX
- **MADE masking**: hidden orders `arange(h_dim) % (N-1)`, input mask `prev_order <= hidden_order`, output mask `hidden_order < output_order`
- **Coupling layers are self-inverse**: `σ_B = z_B * sign(g_φ(z_A))`, so `forward ∘ forward = identity`
- **STE**: `_ste_sign(x) = x + stop_gradient(sign(x) - x)` — forward is sign, backward is identity
- **REINFORCE for θ**: `loss = mean(stop_gradient(advantage) * log_prob)` with EMA baseline
- **STE for φ**: `loss = mean(energy(flow(z)))` with straight-through through sign
- Separate Adam optimizers for θ (base) and φ (flow)
