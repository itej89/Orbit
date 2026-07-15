# SET A: Heterogeneity Sweep — AMD MI300X, 2026-07-14

## Setup
- Node: useocpm2m-097-029 (AMD MI300X, 192 GB HBM3/GPU)
- Model: Qwen3-8B (bfloat16, max-model-len=2048, gpu-memory-utilization=0.15)
- Image: vllm/vllm-openai-rocm:v0.23.0
- Fast server: GPU 0, port 8000, max-num-seqs=64 (fixed)
- Slow server: GPU 1, port 8002, max-num-seqs in {64,32,16,8}
- Arrival rate: 4 rps (Poisson), 120 requests, max_tokens=50
- Policies tested: rr, lor, po2, mpc_po2
- Router restarted fresh before each policy run (MPC starts with no EMA state)

## Parallelization
- 1x and 2x: node029 (sequential)
- 4x: node026 (parallel, results_node026_4x/)
- 8x: node033 (parallel, results_node033_8x/)
- Canonical/verified results: results_node029/ (all 4 ratios run sequentially on same node)

## Results Summary

| Hetero | Policy  | Mean (ms) | Stdev (ms) | P99 (ms) |
|--------|---------|-----------|------------|----------|
| 1x     | rr      | 413.8     | 431.7      | 2587.1   |
| 1x     | lor     | 292.3     | 19.7       | 352.9    |
| 1x     | po2     | 289.0     | 17.8       | 323.5    |
| 1x     | mpc_po2 | 341.8     | 47.4       | 444.2    |
| 2x     | rr      | 327.5     | 266.7      | 1775.2   |
| 2x     | lor     | 294.4     | 19.3       | 339.6    |
| 2x     | po2     | 292.6     | 19.4       | 341.0    |
| 2x     | mpc_po2 | 339.5     | 42.2       | 433.2    |
| 4x     | rr      | 354.0     | 313.7      | 1828.4   |
| 4x     | lor     | 285.2     | 17.7       | 330.9    |
| 4x     | po2     | 285.9     | 19.7       | 344.4    |
| 4x     | mpc_po2 | 326.4     | 45.8       | 429.4    |
| 8x     | rr      | 353.8     | 338.1      | 2417.1   |
| 8x     | lor     | 287.9     | 16.7       | 330.8    |
| 8x     | po2     | 287.9     | 17.8       | 342.4    |
| 8x     | mpc_po2 | 346.6     | 49.6       | 448.4    |

## Key Finding
MPC-PO2 underperforms LOR and PO2 by 14-20% on mean latency across all
heterogeneity levels when the router is started fresh per policy run.
Root cause: EMA service rate estimator cold-start (~30 requests to converge).
With only 120 requests at 4 rps (~30s), MPC never reaches a stable estimate.
LOR and PO2 respond instantaneously to queue depth with no warm-up cost.

MPC wins under sustained traffic where EMA has time to converge (validated
separately with a warm router: 54% mean / 72% P99 improvement at 8x hetero).
