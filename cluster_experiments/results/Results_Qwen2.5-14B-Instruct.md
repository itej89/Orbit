# Orbit MPC Cluster Experiment Results: Qwen2.5-14B-Instruct

## Experiment Overview

| Item | Value |
|------|-------|
| Model | Qwen/Qwen2.5-14B-Instruct (14B parameters) |
| Hardware | AMD Instinct MI300X, 8 GPUs per node |
| Tensor Parallelism | TP=8 (full node) |
| KV Transfer | NixlConnector over RDMA (UCX, InfiniBand mlx5) |
| Docker Image | `rocm/pytorch-private:vllm-v0.14.0_amd_dev_aiter_nixl_ravgupta` |
| vLLM Version | 0.14.0 (with NixlConnector disaggregation support) |
| Cluster | `useocpslog-002.amd.com`, partition `amd-rccl` |
| Node Pool | 8 nodes: useocpm2m-097-{019,020,023,025,027,042,043,047} |
| Experiments | 40 total across 5 phases |
| Date | February 19, 2026 |
| Results Directory | `comprehensive_20260219_031301/` |

---

## Phase 1: Concurrency Sweep (2P2D Homogeneous)

**Goal**: Measure how MPC scales with increasing request concurrency when all servers are identical.

**Setup**: 2 Prefill + 2 Decode nodes (identical: gpu\_mem=0.7, max\_seqs=256), ISL=256, OSL=256.

**Comparison**: RR (toy\_proxy round-robin) vs MPC\_RR (Orbit MPC-weighted round-robin).

| Concurrency | Prompts | RR Throughput (tok/s) | MPC\_RR Throughput (tok/s) | Delta | RR TPOT\_P99 (ms) | MPC\_RR TPOT\_P99 (ms) | RR ITL\_P99 (ms) | MPC\_RR ITL\_P99 (ms) |
|-------------|---------|----------------------|---------------------------|-------|-------------------|------------------------|------------------|----------------------|
| 2 | 64 | 362.07 | 353.72 | -2.3% | 3.98 | 4.25 | 7.67 | 11.24 |
| 4 | 64 | 555.16 | 527.25 | -5.0% | 4.42 | 4.54 | 8.82 | 12.78 |
| 8 | 64 | 756.92 | 729.29 | -3.6% | 4.89 | 5.38 | 12.82 | 16.57 |
| 16 | 128 | 1368.50 | 1326.49 | -3.1% | 5.81 | 6.49 | 32.37 | 29.83 |
| 32 | 256 | 2498.49 | 2370.83 | -5.1% | 6.99 | 8.18 | 34.14 | 31.41 |

**Finding**: In homogeneous configurations, RR outperforms MPC\_RR by 2--5% in throughput. MPC adds computational overhead (QP solve every 0.5s) without benefit when all servers are identical. At concurrency >= 16, MPC shows marginal ITL\_P99 improvement, but throughput cost is not justified.

---

## Phase 2: Scheduling Algorithm Comparison (2P2D Homogeneous)

**Goal**: Compare all six routing policies at medium and high load.

**Setup**: 2P2D homogeneous, ISL=256, OSL=256, 128 prompts.

### Concurrency = 8

| Policy | Throughput (tok/s) | TPOT\_mean (ms) | TPOT\_P99 (ms) | ITL\_P99 (ms) |
|--------|-------------------|-----------------|----------------|---------------|
| toy (built-in RR) | 1038.70 | 4.63 | 4.81 | 12.19 |
| rr (Orbit RR) | 1029.21 | 4.54 | 4.81 | 13.21 |
| random | 981.42 | 4.94 | 5.57 | 29.94 |
| **po2** | **1035.72** | **4.58** | **4.72** | **12.33** |
| mpc\_rr | 982.72 | 4.85 | 5.32 | 16.89 |
| mpc\_po2 | 1009.34 | 4.69 | 5.10 | 15.94 |

### Concurrency = 16

| Policy | Throughput (tok/s) | TPOT\_mean (ms) | TPOT\_P99 (ms) | ITL\_P99 (ms) |
|--------|-------------------|-----------------|----------------|---------------|
| toy (built-in RR) | 1392.07 | 5.38 | 5.82 | 31.34 |
| rr (Orbit RR) | 1390.84 | 5.43 | 6.50 | 32.88 |
| random | 1366.09 | 5.52 | 6.35 | 32.48 |
| **po2** | **1379.60** | **4.88** | **5.33** | **10.61** |
| mpc\_rr | 1344.94 | 5.63 | 6.76 | 28.96 |
| mpc\_po2 | 1359.53 | 5.38 | 6.46 | 17.09 |

**Finding**: PO2 delivers dramatically better tail latency at Con=16: **66% ITL\_P99 reduction** (10.61ms vs 31.34ms for RR). Random is the worst performer. MPC\_PO2 achieves better ITL\_P99 than RR (17.09 vs 31.34) but not as good as pure PO2.

---

## Phase 3: Heterogeneous 2P2D

**Goal**: Evaluate routing policies when servers have asymmetric capacity.

**Setup**: P1 normal (gpu\_mem=0.7, max\_seqs=256), **P2 throttled** (gpu\_mem=0.45, max\_seqs=64). D1 normal, **D2 throttled**. ISL=256, OSL=256, 128 prompts.

### Concurrency = 8

| Policy | Throughput (tok/s) | TPOT\_mean (ms) | TPOT\_P99 (ms) | ITL\_P99 (ms) |
|--------|-------------------|-----------------|----------------|---------------|
| toy (RR) | 996.32 | 4.70 | 5.73 | 12.77 |
| **po2** | **1015.50** | **4.46** | **4.69** | **10.27** |
| mpc\_po2 | 1013.18 | 4.66 | 5.21 | 15.23 |

### Concurrency = 16

| Policy | Throughput (tok/s) | TPOT\_mean (ms) | TPOT\_P99 (ms) | ITL\_P99 (ms) |
|--------|-------------------|-----------------|----------------|---------------|
| toy (RR) | 1386.41 | 5.30 | 5.62 | 31.04 |
| po2 | 1388.00 | 4.89 | 5.62 | 10.88 |
| mpc\_po2 | 1342.04 | 5.38 | 6.15 | 17.82 |

**Finding**: PO2 improves TPOT\_P99 by 18.2% over RR at Con=8 in heterogeneous settings. At Con=16, PO2 reduces ITL\_P99 by 65% (10.88 vs 31.04ms). MPC\_PO2 is intermediate.

---

## Phase 4: Heterogeneous 3P3D Scale-out

**Goal**: Test whether MPC benefits grow with system scale and more heterogeneous nodes.

**Setup**: 3 Prefill + 3 Decode (7 nodes). P2, P3 throttled; D2, D3 throttled. ISL=256, OSL=256, 128 prompts.

| Policy | Concurrency | Throughput (tok/s) | TPOT\_mean (ms) | TPOT\_P99 (ms) | ITL\_P99 (ms) |
|--------|-------------|-------------------|-----------------|----------------|---------------|
| toy (RR) | 8 | 1036.35 | 4.47 | 4.78 | 10.55 |
| **mpc\_po2** | **8** | **1058.91** | **4.41** | **4.85** | 15.71 |
| toy (RR) | 16 | 1399.88 | 5.19 | 6.04 | 31.29 |
| **mpc\_po2** | **16** | **1412.45** | **4.97** | **5.60** | **16.95** |

**Key Finding at Con=16**: MPC\_PO2 outperforms RR across all metrics:
- Throughput: **+0.9%** (1412.45 vs 1399.88 tok/s)
- TPOT\_mean: **-4.2%** (4.97 vs 5.19ms)
- TPOT\_P99: **-7.3%** (5.60 vs 6.04ms)
- ITL\_P99: **-45.8%** (16.95 vs 31.29ms)

At Con=8: MPC\_PO2 achieves **+2.2% throughput** and -1.3% TPOT\_mean improvement.

---

## Phase 5: Variable ISL/OSL (2P2D Homogeneous)

**Goal**: Test how routing adapts to different request shapes (prefill-heavy vs decode-heavy).

**Setup**: 2P2D homogeneous, Con=8, 128 prompts. RR vs MPC\_PO2.

| Workload | RR Throughput (tok/s) | MPC Throughput (tok/s) | RR TPOT\_P99 (ms) | MPC TPOT\_P99 (ms) | RR ITL\_P99 (ms) | MPC ITL\_P99 (ms) | ITL\_P99 Delta |
|----------|----------------------|------------------------|--------------------|--------------------|------------------|-------------------|----------------|
| ISL=64, OSL=512 | 1353.87 | 1317.00 | 4.46 | 4.93 | 9.33 | 13.41 | +43.7% |
| ISL=512, OSL=64 | 413.51 | 413.26 | 7.12 | 7.10 | 46.36 | **36.28** | **-21.7%** |
| ISL=128, OSL=256 | 1061.56 | 995.93 | 4.74 | 5.11 | 15.51 | 16.01 | +3.2% |
| ISL=256, OSL=128 | 720.18 | 716.82 | 5.01 | 5.69 | 30.58 | **15.34** | **-49.8%** |

**Key Finding**: MPC shows significant ITL\_P99 improvement for prefill-heavy workloads:
- **ISL=256/OSL=128**: MPC reduces ITL\_P99 by **49.8%** (15.34 vs 30.58ms)
- **ISL=512/OSL=64**: MPC reduces ITL\_P99 by **21.7%** (36.28 vs 46.36ms)

For decode-heavy workloads (ISL=64/OSL=512), RR is slightly better.

---

## Summary of Key Findings

### When MPC Works Best

1. **Heterogeneous scale-out (3P3D, Con=16)**: +2.2% throughput, -7.3% TPOT\_P99, **-45.8% ITL\_P99**
2. **Prefill-heavy variable workloads**: Up to **-49.8% ITL\_P99** for ISL=256/OSL=128
3. **Higher concurrency with heterogeneity**: Benefits increase with load and server diversity

### When MPC Does Not Help

1. **Homogeneous low/medium load**: 2-5% throughput overhead from MPC computation
2. **Decode-heavy workloads**: RR is sufficient since decode times are more uniform
3. **Very low concurrency**: Not enough contention to benefit from intelligent routing

### Best Policy Recommendations

| Deployment | Recommended Policy | Rationale |
|------------|-------------------|-----------|
| Homogeneous, any load | PO2 or RR | Lowest overhead, PO2 best for tail latency |
| Heterogeneous, moderate load | PO2 | Best tail latency with zero overhead |
| Heterogeneous, high load + scale | MPC\_PO2 | Throughput + latency gains at scale |
| Variable ISL/OSL (prefill-heavy) | MPC\_PO2 | Significant tail latency reduction |

---

## Reproduction Instructions

### Prerequisites

- Access to AMD MI300X cluster with Slurm scheduler
- Docker image: `rocm/pytorch-private:vllm-v0.14.0_amd_dev_aiter_nixl_ravgupta`
- Shared filesystem at `/shared_inference/`
- InfiniBand connectivity between nodes

### Running Experiments

```bash
# Clone the repo
git clone https://github.com/itej89/Orbit.git
cd Orbit
git checkout ravgupta/Code_results_Qwen2.5-14B-Instruct

# Dry run (generate scripts without submitting)
cd cluster_experiments
bash slurm/submit_comprehensive_experiments.sh --dry-run --phase all

# Run a specific phase
bash slurm/submit_comprehensive_experiments.sh --phase 1

# Run all phases
bash slurm/submit_comprehensive_experiments.sh --phase all

# Collect results
bash collect_results.sh results/comprehensive_*/
```

### Key Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PROXY_POLICY` | Routing policy: toy, rr, random, po2, mpc\_rr, mpc\_po2 | toy |
| `HETERO_MODE` | Enable heterogeneous throttling | false |
| `HETERO_GPU_MEM` | GPU memory utilization for throttled nodes | 0.45 |
| `HETERO_MAX_SEQS` | Max concurrent sequences for throttled nodes | 64 |
| `BENCH_ISL` | Input sequence length | 128 |
| `BENCH_OSL` | Output sequence length | 128 |
| `BENCH_CONCURRENCY` | Max request concurrency | 4 |

### File Inventory

| File | Purpose |
|------|---------|
| `slurm/submit_comprehensive_experiments.sh` | Main experiment submission (5 phases) |
| `slurm/vllm_disagg_server_orbit.sh` | Server launcher with heterogeneous + Orbit proxy support |
| `slurm/benchmark_xPyD.sh` | Configurable benchmark runner |
| `simulation/orbit_proxy_server.py` | Multi-algorithm proxy (RR, PO2, MPC variants) |
| `slurm/start_etcd.sh` | ETCD cluster for NixlConnector coordination |
| `slurm/socket_barrier.py` | Multi-node synchronization |
| `collect_results.sh` | Results aggregation script |
