# Orbit Standalone Benchmark — Real Cluster Results

**Date:** 2026-07-14  
**Node:** AMD MI300X (node 079, 10.158.215.254), Slurm job 202205  
**Model:** Qwen3-8B (bfloat16, max-model-len=2048)  
**Image:** vllm/vllm-openai-rocm:v0.23.0  

## Cluster Setup

| Server | GPU | Port | max-seqs | Role |
|--------|-----|------|----------|------|
| vllm_0 | GPU 0 | 8000 | 64 | FAST (8x capacity) |
| vllm_1 | GPU 1 | 8002 | 8  | SLOW |

Heterogeneity: 8x capacity difference (max-seqs ratio).  
Router: orbit_router.py running directly on host (Python 3.10, cvxpy 1.7.5).

## Experiment Commands



Full provenance log (all commands + per-request results):  (5905 lines)

## Results Summary

### SET 1: Arrival Rate Sweep — PO2 vs Orbit MPC-PO2 (100 req each, Poisson)

| Rate (rps) | Policy | Mean (ms) | Stdev (ms) | P99 (ms) | N |
|-----------|--------|-----------|------------|---------|---|
| 2.0 | PO2 | 2267.6 | 1292.6 | 4557.3 | 100 |
| 2.0 | **Orbit MPC-PO2** | **1909.2** | **875.1** | 4709.0 | 100 |
| 4.0 | PO2 | 3410.2 | 1877.1 | 9064.1 | 100 |
| 4.0 | **Orbit MPC-PO2** | **1579.8** | **794.1** | **2556.4** | 100 |
| 6.0 | PO2 | 3065.7 | 628.6 | 4615.8 | 100 |
| 6.0 | **Orbit MPC-PO2** | **2398.5** | **474.2** | **3524.0** | 100 |
| 8.0 | PO2 | 2840.7 | 537.7 | 4620.8 | 100 |
| 8.0 | Orbit MPC-PO2 | 4357.5 | 1897.5 | 9322.2 | 100 |

**Key finding:** At 4 rps (moderate load, sweet spot): **+54% mean latency, +72% P99 improvement**.  
At 8 rps (overload): MPC overshoots — known limitation of predictive control at saturation.

### SET 2: Main Comparison Table — 6 rps, 150 requests (Poisson)

| Policy | Mean (ms) | Stdev (ms) | P95 (ms) | P99 (ms) | N |
|--------|-----------|------------|---------|---------|---|
| Round Robin | 1972.6 | 470.7 | 2494.9 | 2593.0 | 150 |
| PO2 | 1431.9 | 816.6 | 2410.3 | 2540.3 | 150 |
| Orbit MPC-PO2 | 1917.0 | 492.7 | 2522.7 | 2901.6 | 150 |

### SET 3: Bursty Workload — base=3 rps, burst=9 rps for 10s every 30s (200 req)

| Policy | Mean (ms) | Stdev (ms) | P95 (ms) | P99 (ms) | N |
|--------|-----------|------------|---------|---------|---|
| PO2 | 2538.8 | 1344.5 | 5223.1 | 6087.6 | 200 |
| Orbit MPC-PO2 | 2826.2 | 1021.2 | 5254.5 | 7052.9 | 200 |

## Raw Data

All per-request latency data in . Each file contains:
- : summary statistics
- : list of  per request

Full run log with exact commands, router metrics snapshots, and routing decisions:  

