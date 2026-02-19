# Orbit MPC Experiment Plan
## Disaggregated Prefill/Decode with Heterogeneous GPU Allocation

---

## 1. Architecture Overview

### Base Setup (from PR #134)
```
┌─────────────────────────────────────────────────────────────────┐
│                    Disaggregated vLLM Architecture              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐                                               │
│   │   Proxy     │◄── Client Requests                            │
│   │  (Node 0)   │                                               │
│   │ Orbit MPC   │                                               │
│   └──────┬──────┘                                               │
│          │                                                      │
│    ┌─────┴─────┐                                                │
│    ▼           ▼                                                │
│ ┌──────────┐ ┌──────────┐    ┌──────────┐ ┌──────────┐         │
│ │ Prefill  │ │ Prefill  │    │ Decode   │ │ Decode   │         │
│ │  Node 1  │ │  Node 2  │    │  Node 3  │ │  Node 4  │         │
│ │  TP=X    │ │  TP=Y    │    │  TP=X    │ │  TP=Y    │         │
│ │ (GPUs)   │ │ (GPUs)   │    │ (GPUs)   │ │ (GPUs)   │         │
│ └──────────┘ └──────────┘    └──────────┘ └──────────┘         │
│      KV Producer              KV Consumer                       │
│      NixlConnector            NixlConnector                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Orbit MPC Integration Point
- Replace `toy_proxy_server.py` with **Orbit MPC Router**
- Router receives metrics from prefill/decode servers
- MPC computes optimal routing weights based on:
  - Queue depths at each server
  - Service rates (varying with TP configuration)
  - Predicted load trajectory

---

## 2. Models of Interest

| Model | Size | TP Options | Memory/GPU | Use Case |
|-------|------|------------|------------|----------|
| **Qwen2.5-14B-Instruct** | 14B | TP=2,4,8 | ~7GB@TP=4 | Primary test model |
| **Qwen2.5-32B-Instruct** | 32B | TP=2,4,8 | ~16GB@TP=4 | Medium scale |
| **Llama-3.3-70B-Instruct-FP8** | 70B | TP=4,8 | ~18GB@TP=8 | Production scale |

---

## 3. Heterogeneity Configurations

### Single Physical Server (8 GPUs MI300X)
**Goal**: Fit multiple P/D services with varying TP on one node

| Config | Services per Node | GPU Allocation | Expected Behavior |
|--------|-------------------|----------------|-------------------|
| **2P_TP4** | 2 Prefill | GPUs 0-3, 4-7 | Equal capacity |
| **2P_TP2_TP4** | 2 Prefill | GPUs 0-1 (TP2), 2-5 (TP4) | 2x heterogeneity |
| **3P_TP2** | 3 Prefill | GPUs 0-1, 2-3, 4-5 | 3 services, 6 GPUs |
| **4P_TP2** | 4 Prefill | GPUs 0-1, 2-3, 4-5, 6-7 | 4 services, 8 GPUs |

### Multi-Node Configurations

| Config | Nodes | Prefill | Decode | Description |
|--------|-------|---------|--------|-------------|
| **1P1D** | 3 | 1×TP8 | 1×TP8 | Baseline homogeneous |
| **2P2D_homo** | 5 | 2×TP8 | 2×TP8 | Homogeneous scale-out |
| **2P2D_het** | 5 | 1×TP4+1×TP8 | 1×TP4+1×TP8 | Heterogeneous |
| **4P4D_homo** | 9 | 4×TP8 | 4×TP8 | Large homogeneous |
| **4P4D_het** | 9 | 2×TP4+2×TP8 | 2×TP4+2×TP8 | Large heterogeneous |

### GPU-Packed Configurations (Multiple Services per Node)
**Using Docker + HIP_VISIBLE_DEVICES to pack services**

| Config | Physical Nodes | Services | GPU Mapping |
|--------|----------------|----------|-------------|
| **packed_2P2D** | 2 nodes | 4 services | Node1: 2 Prefill (TP2+TP4), Node2: 2 Decode (TP2+TP4) |
| **packed_4P4D** | 4 nodes | 8 services | 2 services per node |

---

## 4. MPC Benefit Categories

### Where MPC Helps Most
1. **Heterogeneous Capacity** (TP2 vs TP4 vs TP8)
   - Different TP = different throughput
   - MPC adapts routing weights dynamically

2. **Variable Workload** (ISL/OSL variance)
   - Long prompts → more prefill work
   - Long outputs → more decode work
   - MPC predicts and balances

3. **Bursty Traffic**
   - Sudden request spikes
   - MPC smooths queue depths

4. **Mixed Model Sizes** (future)
   - Different models on different servers
   - MPC handles capacity differences

### Where MPC Adds Little Value
1. **Homogeneous TP8 configurations** (all servers equal)
2. **Very low load** (no queuing)
3. **Uniform workload** (steady, predictable)

---

## 5. Experiment Matrix

### Phase 1: Baseline Validation (Homogeneous)
| Experiment | Config | Model | Workload | Expected |
|------------|--------|-------|----------|----------|
| P1.1 | 1P1D_TP8 | Qwen-14B | uniform | Baseline perf |
| P1.2 | 2P2D_TP8 | Qwen-14B | uniform | Scale-out baseline |

### Phase 2: Heterogeneous (MPC Target)
| Experiment | Config | Model | Heterogeneity | Expected MPC Benefit |
|------------|--------|-------|---------------|---------------------|
| P2.1 | 2P2D_het | Qwen-14B | TP4+TP8 | 10-20% P99 improvement |
| P2.2 | 2P2D_het | Qwen-14B | TP2+TP8 | 20-30% P99 improvement |
| P2.3 | 4P4D_het | Qwen-14B | mixed | Scale-out heterogeneous |

### Phase 3: Workload Sensitivity
| Experiment | Config | Workload Pattern | MPC Focus |
|------------|--------|------------------|-----------|
| P3.1 | 2P2D_het | variable_osl (10-500) | Output variance |
| P3.2 | 2P2D_het | bursty (spike pattern) | Queue management |
| P3.3 | 2P2D_het | ramping | Adaptation speed |

### Phase 4: GPU-Packed (Dense Deployment)
| Experiment | Physical Nodes | Services | Description |
|------------|----------------|----------|-------------|
| P4.1 | 2 | 4 (2P2D) | 2 services per node |
| P4.2 | 4 | 8 (4P4D) | 2 services per node |

---

## 6. Metrics to Collect

| Metric | Description | Target |
|--------|-------------|--------|
| **Throughput** | Requests/sec | Higher is better |
| **Mean Latency** | Average E2E | Lower is better |
| **P99 Latency** | Tail latency | Primary MPC target |
| **Std Dev** | Consistency | Lower = more stable |
| **Queue Imbalance** | Max-Min queue depth | MPC should reduce |
| **MPC Weights** | Routing distribution | Should adapt to load |

---

## 7. Implementation Steps

### Step 1: Adapt Proxy Server
- Modify `toy_proxy_server.py` to include MPC logic
- Add metrics collection endpoint
- Implement weighted routing

### Step 2: GPU Allocation via Docker
```bash
# Example: 2 services on one node
# Service 1: TP=2 on GPUs 0,1
docker run -e HIP_VISIBLE_DEVICES=0,1 --name prefill_tp2 ...

# Service 2: TP=4 on GPUs 2,3,4,5
docker run -e HIP_VISIBLE_DEVICES=2,3,4,5 --name prefill_tp4 ...
```

### Step 3: Slurm Job Templates
- Create templates for each configuration
- Support both homogeneous and heterogeneous
- Include MPC enable/disable flag

### Step 4: Automated Result Collection
- Parse logs for metrics
- Generate comparison tables
- Create visualization plots

---

## 8. Expected Outcomes

| Scenario | Baseline | MPC | Expected Improvement |
|----------|----------|-----|---------------------|
| Homogeneous | Good | Similar | <5% (not target) |
| 2x Hetero (TP4+TP8) | Moderate | Better | 10-20% P99 |
| 4x Hetero (TP2+TP8) | Variable | Much Better | 20-35% P99 |
| Bursty Load | High variance | Stable | 15-25% P99, lower std |

---

## 9. Timeline

| Phase | Tasks | Duration |
|-------|-------|----------|
| Phase 1 | Baseline validation | 2-4 hours |
| Phase 2 | Heterogeneous experiments | 4-8 hours |
| Phase 3 | Workload sensitivity | 2-4 hours |
| Phase 4 | GPU-packed experiments | 4-8 hours |
| Analysis | Results processing | 2-4 hours |

---

## 10. Key Files to Create

1. `orbit_proxy_server.py` - MPC-enabled proxy replacing toy_proxy_server
2. `submit_disagg_orbit.sh` - Slurm submission for disagg experiments
3. `collect_results.py` - Automated result parsing
4. `generate_figures.py` - Paper figure generation
