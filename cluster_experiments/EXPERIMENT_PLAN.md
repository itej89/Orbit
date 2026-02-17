# Orbit Cluster Experiment Plan

## Overview

This document outlines the experimental plan for validating Orbit MPC on real vLLM disaggregated serving workloads using the AMD cluster.

## Cluster Access

- **Login Node:** `useocpslog-002.amd.com`
- **Important:** Do NOT run Docker images or heavy workloads on the login node. Submit jobs to compute nodes.

## Docker Images

| Image | Description |
|-------|-------------|
| `rocm/vllm:v0.14.0_amd_dev` | Base upstream vLLM image |
| `rocm/pytorch-private:vllm-v0.14.0_amd_dev_aiter_nixl_ravgupta` | Completed image with AITER+NIXL |
| `rocm/pytorch-private:vllm-v0.14.0_orbit_mpc` | **To Build:** Image with Orbit MPC router |

## Models to Test

| Model | Type | Size | TP Config | Notes |
|-------|------|------|-----------|-------|
| `meta-llama/Llama-3.1-70B-Instruct` | Dense | 70B | TP=8 | Baseline dense model |
| `Qwen/Qwen1.5-MoE-A2.7B` | MoE | 14.3B (2.7B active) | TP=2,4 | Small MoE |
| `allenai/OLMoE-1B-7B-0924` | MoE | 7B (1B active) | TP=2 | Small MoE |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | MoE | 47B | TP=4,8 | Medium MoE |
| `Qwen/Qwen2-VL-7B-Instruct` | Vision+MoE | 7B | TP=2,4 | Vision model |

## Server Configurations

### Homogeneous Configurations

All prefill and decode servers use the same TP/EP settings.

| Config | Prefill Servers | Decode Servers | Total GPUs |
|--------|-----------------|----------------|------------|
| 2P2D-TP8 | 2 × TP=8 | 2 × TP=8 | 32 |
| 4P4D-TP8 | 4 × TP=8 | 4 × TP=8 | 64 |
| 2P2D-TP4 | 2 × TP=4 | 2 × TP=4 | 16 |

### Heterogeneous Configurations (Key Focus)

Mixed TP settings to create capacity asymmetry - this is where MPC should show benefit.

| Config | Prefill Servers | Decode Servers | Notes |
|--------|-----------------|----------------|-------|
| HET-2P2D-A | P1:TP=8, P2:TP=4 | D1:TP=8, D2:TP=4 | 2× capacity diff |
| HET-2P2D-B | P1:TP=8, P2:TP=2 | D1:TP=8, D2:TP=2 | 4× capacity diff |
| HET-4P4D | P1-2:TP=8, P3-4:TP=4 | D1-2:TP=8, D3-4:TP=4 | Mixed pool |
| HET-MIXED | P1:TP=8, P2:TP=4, P3:TP=2 | D1:TP=8, D2:TP=4 | Highly varied |

## GPU Allocation Strategy

Use `HIP_VISIBLE_DEVICES` to control GPU assignment and avoid contention.

```bash
# Example: 8-GPU node, running TP=4 prefill and TP=4 decode
# Prefill server (GPUs 0-3)
HIP_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server ...

# Decode server (GPUs 4-7)
HIP_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server ...
```

### Multi-Node Example

```bash
# Node 1: Prefill servers
# Server P1 (TP=8, all GPUs)
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 8 \
    --port 8100

# Node 2: Prefill P2 (TP=4, first 4 GPUs) + Decode D1 (TP=4, last 4 GPUs)
# Terminal 1:
HIP_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --port 8101

# Terminal 2:
HIP_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --port 8200
```

## Routing Policies to Compare

| Policy | Description | MPC Enabled |
|--------|-------------|-------------|
| `round_robin` | Baseline cyclic | No |
| `random` | Baseline random | No |
| `power_of_two` | vLLM default PO2 | No |
| `cache_aware` | vLLM cache-aware | No |
| `mpc_round_robin` | Orbit MPC + RR | Yes |
| `mpc_power_of_two` | Orbit MPC + PO2 | Yes |

## Workload Configurations

### Synthetic Workloads

| Workload | Prompt Length | Output Length | Arrival Pattern |
|----------|---------------|---------------|-----------------|
| SHORT | 128-256 tokens | 32-64 tokens | Poisson λ=10 |
| MEDIUM | 512-1024 tokens | 64-256 tokens | Poisson λ=5 |
| LONG | 2048-4096 tokens | 256-512 tokens | Poisson λ=2 |
| MIXED | Variable | Variable | Bursty |

### Real-World Traces (Optional)

- ShareGPT conversation traces
- Code completion patterns

## Metrics to Collect

### Latency Metrics
- Mean, P50, P90, P95, P99 end-to-end latency
- Time-to-First-Token (TTFT)
- Inter-Token Latency (ITL)

### Throughput Metrics
- Requests per second (sustained)
- Tokens per second (input + output)
- Throughput variance over time

### System Metrics
- GPU utilization per server
- Memory utilization
- Queue depths over time
- MPC weight trajectories

## Experiment Matrix

### Phase 1: Homogeneous Validation
Verify MPC doesn't hurt performance when servers are equal.

| Model | Config | Policies | Expected Result |
|-------|--------|----------|-----------------|
| Llama-70B | 2P2D-TP8 | PO2, MPC-PO2 | Similar performance |
| Mixtral-8x7B | 2P2D-TP8 | PO2, MPC-PO2 | Similar performance |

### Phase 2: Heterogeneous Evaluation (Key)
Validate MPC improves load balancing under capacity asymmetry.

| Model | Config | Policies | Expected Result |
|-------|--------|----------|-----------------|
| Llama-70B | HET-2P2D-A | All | MPC > baselines |
| Llama-70B | HET-2P2D-B | All | MPC >> baselines |
| Mixtral-8x7B | HET-4P4D | All | MPC > baselines |
| OLMoE-1B-7B | HET-2P2D-A | All | MPC > baselines |

### Phase 3: Scale-Out
Test at larger scale.

| Model | Config | Policies | Notes |
|-------|--------|----------|-------|
| Llama-70B | 4P4D-TP8 | PO2, MPC-PO2 | 64 GPUs |
| Llama-70B | HET-4P4D | All | Heterogeneous scale |

## Scripts to Create

1. `setup_cluster_env.sh` - Environment setup
2. `start_vllm_servers.sh` - Launch vLLM servers with configs
3. `start_orbit_router.sh` - Launch Orbit router
4. `run_benchmark.sh` - Run workload and collect metrics
5. `collect_results.sh` - Aggregate results
6. `plot_results.py` - Generate figures for paper

## Directory Structure

```
cluster_experiments/
├── EXPERIMENT_PLAN.md          # This file
├── configs/
│   ├── models.yaml             # Model configurations
│   ├── servers.yaml            # Server configurations
│   └── workloads.yaml          # Workload definitions
├── scripts/
│   ├── setup_cluster_env.sh
│   ├── start_vllm_servers.sh
│   ├── start_orbit_router.sh
│   ├── run_benchmark.sh
│   └── collect_results.sh
├── results/
│   └── (experiment outputs)
└── analysis/
    └── plot_results.py
```

## Timeline

1. **Week 1:** Set up Docker images, validate basic vLLM disagg
2. **Week 2:** Integrate MPC router, test on homogeneous config
3. **Week 3:** Run heterogeneous experiments
4. **Week 4:** Analysis, paper updates, scale-out tests

## Notes

### vLLM Disaggregation Setup

From the MAD-private repo (`ravgupta/add_vllm_router` branch):

```bash
# Modified vLLM server args for disagg
["DeepSeek-V3"]="--tensor-parallel-size 8 --compilation-config '{\"cudagraph_mode\":\"PIECEWISE\"}' --no-enable-prefix-caching --block-size 1"
```

### Known Issues

- `full_cuda_graph` not valid in vLLM 0.14.0rc3 - removed from compilation-config
- Need NIXL connector for prefill-decode transfer
