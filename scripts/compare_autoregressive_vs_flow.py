#!/usr/bin/env python
"""Compare autoregressive-only (large MADE) vs flow-based model on 2D Ising."""

import jax
import jax.numpy as jnp

from dsflow_ising.config import ModelConfig, TrainConfig
from dsflow_ising.train import train
from dsflow_ising.diagnostics import variational_free_energy, base_entropy

L = 8
N = L * L
NUM_STEPS = 2000
T = 2.269

# --- Run 1: Autoregressive only, large MADE (hidden_dim=1024) ---
print("=" * 60)
print("Run 1: Autoregressive only (no flow, MADE hidden=1024)")
print("=" * 60)

model_cfg_ar = ModelConfig(L=L, n_flow_layers=0, made_hidden_dim=1024)
train_cfg_ar = TrainConfig(T=T, num_steps=NUM_STEPS, seed=42)

state_ar, hist_ar, made_ar, flow_ar, pairs = train(
    model_cfg_ar, train_cfg_ar, log_every=200,
    log_file="logs/autoregressive_only.csv",
)

# --- Run 2: Flow-based (4 layers, default MADE) ---
print("\n" + "=" * 60)
print("Run 2: Flow-based (4 layers, MADE hidden=256)")
print("=" * 60)

model_cfg_flow = ModelConfig(L=L, n_flow_layers=4, mask_features=(16, 16))
train_cfg_flow = TrainConfig(T=T, num_steps=NUM_STEPS, seed=42)

state_flow, hist_flow, made_flow, flow_flow, _ = train(
    model_cfg_flow, train_cfg_flow, log_every=200,
    log_file="logs/flow_based.csv",
)

# --- Comparison ---
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

key = jax.random.PRNGKey(9999)
k1, k2, k3, k4 = jax.random.split(key, 4)

f_ar = variational_free_energy(
    made_ar, state_ar.made_params, flow_ar, state_ar.flow_params,
    pairs, 1.0, T, k1, num_samples=2000,
)
f_flow = variational_free_energy(
    made_flow, state_flow.made_params, flow_flow, state_flow.flow_params,
    pairs, 1.0, T, k2, num_samples=2000,
)
h_ar = base_entropy(made_ar, state_ar.made_params, k3, num_samples=2000)
h_flow = base_entropy(made_flow, state_flow.made_params, k4, num_samples=2000)

print(f"\n{'Metric':<25} {'Autoregressive':>15} {'Flow-based':>15}")
print("-" * 55)
print(f"{'F_var':<25} {float(f_ar):>15.4f} {float(f_flow):>15.4f}")
print(f"{'F_var / N':<25} {float(f_ar)/N:>15.4f} {float(f_flow)/N:>15.4f}")
print(f"{'Base entropy H':<25} {float(h_ar):>15.4f} {float(h_flow):>15.4f}")
print(f"{'H / N ln2':<25} {float(h_ar)/(N*jnp.log(2)):>15.4f} {float(h_flow)/(N*jnp.log(2)):>15.4f}")

# Training curves: first and last values
print(f"\n{'F_var (step 1)':<25} {float(hist_ar[0]['f_var']):>15.4f} {float(hist_flow[0]['f_var']):>15.4f}")
print(f"{'F_var (final)':<25} {float(hist_ar[-1]['f_var']):>15.4f} {float(hist_flow[-1]['f_var']):>15.4f}")

print(f"\nLower F_var is better (tighter variational bound).")
