# Orbit MPC - Cluster Experiments

Model Predictive Controller Augmentation for Standard Routing Algorithms in Disaggregated vLLM Serving.

## Overview

This directory contains all scripts for running Orbit MPC experiments on the AMD MI300X cluster. The experiments evaluate MPC-augmented load balancing for disaggregated vLLM (prefill/decode separation with NixlConnector KV-cache transfer).

## Directory Structure

```
cluster_experiments/
├── slurm/
│   ├── vllm_disagg_server_orbit.sh        # Main server launcher (runs inside Docker)
│   ├── submit_comprehensive_experiments.sh # Comprehensive experiment submission
│   ├── submit_disagg_orbit.sh             # Legacy experiment submission
│   ├── orbit_proxy_server.py              # → symlinked from simulation/
│   ├── benchmark_xPyD.sh                  # Benchmark runner (configurable ISL/OSL/CON)
│   ├── socket_barrier.py                  # Node synchronization barrier
│   ├── socket_wait.py                     # Wait for socket close
│   ├── start_etcd.sh                      # ETCD server launcher
│   └── port_allocator.py                  # Dynamic port allocation
├── results/                               # Experiment results (auto-generated)
│   └── comprehensive_YYYYMMDD_HHMMSS/
│       ├── MANIFEST.txt                   # Experiment description
│       ├── JOBS.txt                       # Submitted job IDs
│       ├── scripts/                       # Generated sbatch scripts
│       ├── logs/                          # Slurm stdout/stderr logs
│       └── <job_name>/                    # Per-job benchmark results
└── README.md
```

## Prerequisites

- Docker image: `rocm/pytorch-private:vllm-v0.14.0_amd_dev_aiter_nixl_ravgupta`
- Cluster: `useocpslog-002.amd.com` with Slurm scheduler
- Partition: `amd-rccl`
- Node pool: 8 MI300X nodes (8 GPUs each)
- Shared filesystem: `/shared_inference/ravgupta/`

## Docker Image

Based on `Dockerfile.pr134` from the MAD-private PR#134 disaggregated vLLM setup. Located at:
```
/shared_inference/ravgupta/vllm_disagg_testing/Dockerfile.pr134
```

## Key Scripts

### `vllm_disagg_server_orbit.sh`

Runs inside each Docker container. Handles:
- Dynamic port allocation across nodes (avoids conflicts from previous runs)
- ETCD cluster setup for NixlConnector coordination
- Model-specific configuration (TP size, KV cache, max model len)
- Heterogeneous mode: throttles alternating P/D nodes via `--gpu-memory-utilization` and `--max-num-seqs`
- Proxy selection: toy (vLLM built-in RR) or orbit proxy with configurable policy

**Environment variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | required | HuggingFace model path |
| `MODEL_NAME` | required | Config map key (e.g., `Qwen14B`) |
| `xP` | 1 | Number of prefill nodes |
| `yD` | 1 | Number of decode nodes |
| `PROXY_POLICY` | `toy` | Routing: toy, rr, random, po2, mpc_rr, mpc_po2 |
| `HETERO_MODE` | `false` | Enable heterogeneous node throttling |
| `HETERO_GPU_MEM` | `0.45` | GPU memory util for throttled nodes |
| `HETERO_MAX_SEQS` | `64` | Max concurrent sequences for throttled nodes |
| `BENCH_ISL` | `128` | Input sequence length for benchmark |
| `BENCH_OSL` | `128` | Output sequence length for benchmark |
| `BENCH_CONCURRENCY` | `4` | Max concurrency (comma-separated for sweep) |
| `BENCH_PROMPTS` | `32` | Number of benchmark prompts |

### `orbit_proxy_server.py`

Drop-in replacement for vLLM's `toy_proxy_server.py`. Follows the same two-phase protocol (prefill with `do_remote_decode=True` → decode with `kv_transfer_params`).

**Routing policies:**
- `rr` - Round-robin (equivalent to toy_proxy_server)
- `random` - Random server selection
- `po2` - Power-of-Two-Choices (picks 2 random, routes to less loaded)
- `mpc_rr` - MPC-weighted probabilistic round-robin
- `mpc_po2` - MPC-augmented Power-of-Two-Choices (flagship policy)

MPC uses `cvxpy` with OSQP solver to compute optimal routing weights based on:
- Per-server queue depths
- Estimated service rates
- Arrival rate estimation (EWMA)
- Regularization for smooth weight transitions

### `benchmark_xPyD.sh`

Runs `vllm bench serve` with configurable parameters. Supports comma-separated concurrency levels for sweep experiments.

## Running Experiments

### Full experiment suite (recommended)

```bash
# Dry run (generate scripts without submitting)
./slurm/submit_comprehensive_experiments.sh --dry-run --phase all

# Run phase 1 only (concurrency sweep)
./slurm/submit_comprehensive_experiments.sh --phase 1

# Run all phases sequentially (waits for each job)
./slurm/submit_comprehensive_experiments.sh --phase all --sequential

# Run all phases (submit all, let Slurm manage)
./slurm/submit_comprehensive_experiments.sh --phase all
```

### Experiment Phases

| Phase | Config | Description | Jobs |
|-------|--------|-------------|------|
| 1 | 2P2D homo | Concurrency sweep (con=2,4,8,16,32) with RR vs MPC_RR | 10 |
| 2 | 2P2D homo | All 6 scheduling policies at con=8 and con=16 | 12 |
| 3 | 2P2D hetero | Throttled nodes with toy vs po2 vs mpc_po2 | 6 |
| 4 | 3P3D hetero | Scale-out with toy vs mpc_po2 | 4 |
| 5 | 2P2D homo | Variable ISL/OSL patterns | 8 |

### Node Allocation

- **2P2D** (5 nodes): Proxy=019, P1=020, P2=023, D1=025, D2=027
- **3P3D** (7 nodes): Proxy=019, P1=020, P2=023, P3=025, D1=027, D2=042, D3=043

## Analyzing Results

Benchmark logs are in `results/<experiment>/logs/` and per-job directories. Each benchmark log contains:
- Throughput (req/s, tokens/s)
- Latency statistics (mean, median, P99 TTFT, TPOT, ITL, E2E)
- Success rate

Key metrics to compare:
- **Throughput (tok/s)**: Higher is better
- **P99 TPOT**: Lower is better (tail latency for time-per-output-token)
- **P99 E2E**: Lower is better (end-to-end request latency)

## Heterogeneous Mode

In heterogeneous mode, alternating P/D nodes are throttled:
- Normal nodes: `gpu_mem=0.7`, `max_seqs=256`
- Throttled nodes: `gpu_mem=0.45`, `max_seqs=64`

This creates ~2x capacity asymmetry, simulating real-world scenarios where servers have different hardware or competing workloads. RR distributes load equally (sub-optimal), while MPC adapts routing weights to match actual server capacity.
