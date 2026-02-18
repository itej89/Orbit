# Orbit MPC Cluster Experiment Plan

## Overview

This document outlines the comprehensive experiment plan for evaluating Orbit's MPC-augmented 
routing on real vLLM disaggregated serving clusters.

## Goals

1. **Validate simulation findings**: Confirm that MPC provides 15-16% improvement with 2-4× heterogeneity
2. **Test variable ISL/OSL**: Prove MPC handles real-world input/output length distributions
3. **Scale testing**: Verify MPC benefits hold across 2, 4, 8 GPU configurations
4. **Model diversity**: Test across different model architectures (dense, MoE)

## Cluster Configuration

- **Cluster**: useocpslog-002.amd.com
- **Partition**: amd-rccl (34 idle nodes available)
- **GPUs per node**: 8x AMD MI250X
- **Docker image**: `rocm/pytorch-private:vllm-v0.14.0_orbit_mpc`

## Experiment Matrix

### Server Configurations (Heterogeneity Testing)

| Config Name | Prefill TP | Decode TP | Total GPUs | Heterogeneity Ratio |
|-------------|------------|-----------|------------|---------------------|
| homo_8x8    | 8          | 8         | 16         | 1.0× (baseline)     |
| het_8x4     | 8          | 4         | 12         | 2.0×                |
| het_8x2     | 8          | 2         | 10         | 4.0×                |
| het_4x2     | 4          | 2         | 6          | 2.0×                |
| het_2p2d    | 2×4        | 2×2       | 12         | Multi-instance      |

**Rationale**: Different TP configurations create natural heterogeneity:
- Higher TP = lower per-GPU memory pressure but more communication overhead
- Lower TP = higher memory pressure but faster small batch processing
- This creates the capacity asymmetry that MPC is designed to handle

### Workload Patterns (Variable ISL/OSL)

| Pattern Name    | ISL Range   | OSL Range   | Description                    |
|-----------------|-------------|-------------|--------------------------------|
| uniform_short   | 100-500     | 20-50       | Chatbot queries, short answers |
| uniform_long    | 100-500     | 100-500     | Longer conversations           |
| variable_isl    | 50-2000     | 50-100      | Document Q&A, varied context   |
| variable_osl    | 200-400     | 10-500      | Code completion, varied output |
| high_variance   | 50-3000     | 10-1000     | Mixed workload (stress test)   |

**Why Variable ISL/OSL Matters**:
- Prefill time scales with ISL (compute-bound)
- Decode time scales with OSL × batch_size (memory-bound)
- Static routing doesn't adapt to these dynamics
- MPC predicts queue evolution based on observed rates

### Models to Test

| Model | Type | Size | Why Test |
|-------|------|------|----------|
| Llama-3.1-70B-Instruct | Dense | 70B | Production workload baseline |
| Qwen1.5-MoE-A2.7B | MoE | 2.7B active | Expert routing dynamics |
| OLMoE-1B-7B | MoE | 1B active | Smaller MoE variant |
| Mixtral-8x7B | MoE | 8×7B | Popular MoE baseline |

## Experiment Schedule

### Phase 1: Baseline Validation (Day 1)
```
homo_8x8 × uniform_short × [baseline, mpc] × Llama-70B
het_8x4  × uniform_short × [baseline, mpc] × Llama-70B
het_8x2  × uniform_short × [baseline, mpc] × Llama-70B
```
**Expected**: Confirm homogeneous overhead, heterogeneous improvement

### Phase 2: ISL/OSL Variance (Day 1-2)
```
het_8x4 × [all workloads] × [baseline, mpc] × Llama-70B
```
**Expected**: MPC maintains advantage across workload patterns

### Phase 3: Scale Testing (Day 2)
```
[all configs] × variable_osl × [baseline, mpc] × Llama-70B
```
**Expected**: Verify benefits across GPU configurations

### Phase 4: Model Diversity (Day 3)
```
het_8x4 × uniform_short × [baseline, mpc] × [all models]
```
**Expected**: MPC works across model architectures

## Metrics Collected

### Latency
- Mean end-to-end latency (ms)
- P50, P90, P95, P99 latency (ms)
- Standard deviation (variance indicator)

### Throughput
- Requests per second achieved
- Tokens per second (total)
- Success rate (%)

### Time-to-First-Token (TTFT)
- Mean TTFT (ms)
- P99 TTFT (ms)
- Critical for interactive applications

### System Metrics
- GPU utilization per server
- Queue depths over time
- MPC weight evolution (when enabled)

## Expected Results

Based on simulation findings:

| Configuration | MPC Mean Improvement | MPC P99 Improvement |
|---------------|---------------------|---------------------|
| Homogeneous   | -5% to -10% (overhead) | ~0%              |
| 2× Het.       | +12% to +18%        | +5% to +15%         |
| 4× Het.       | +10% to +16%        | +20% to +40%        |
| Bursty load   | +0% to +5%          | +30% to +40%        |

## Running Experiments

### Quick Test (Single Configuration)
```bash
ssh ravgupta@useocpslog-002.amd.com
cd /path/to/orbit_paper/cluster_experiments/slurm
./submit_orbit_experiments.sh --quick --dry-run  # Preview
./submit_orbit_experiments.sh --quick             # Submit
```

### Full Matrix
```bash
./submit_orbit_experiments.sh --dry-run  # Preview all jobs
./submit_orbit_experiments.sh            # Submit all jobs
```

### Monitor Progress
```bash
squeue -u $USER
tail -f results/*/logs/*.out
```

### Analyze Results
```bash
python ../analysis/analyze_cluster_results.py results/<timestamp>/
```

## Deliverables

1. **Raw results**: JSON metrics for each experiment
2. **Comparison tables**: MPC vs baseline across all configurations
3. **Figures**: 
   - Latency CDFs by heterogeneity level
   - MPC improvement vs heterogeneity ratio
   - Throughput stability (time series)
4. **Paper updates**: Final numerical results for JSSPP paper

## Timeline

- **Day 1**: Phases 1-2, initial validation
- **Day 2**: Phase 3, scale testing
- **Day 3**: Phase 4, model diversity + analysis
- **Day 4**: Paper updates with real cluster results

## Notes

- Each experiment runs for 120 seconds of benchmarking
- Allow 10-15 minutes per job (startup + warmup + benchmark + cleanup)
- Full matrix: ~50 experiments × 15 min = ~12 hours total
- Use `--quick` flag for rapid iteration during debugging
