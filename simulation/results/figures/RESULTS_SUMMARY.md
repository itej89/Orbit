# Orbit Simulation Results Summary

## Key Findings

### MPC Improvement by Heterogeneity Level

| Configuration | Baseline Mean (ms) | MPC Mean (ms) | Improvement |
|--------------|-------------------|---------------|-------------|
| homogeneous | 779.4 | 848.2 | -8.8% |
| het_2x | 526.8 | 444.2 | +15.7% |
| het_4x | 658.9 | 551.9 | +16.2% |
| het_6x | 773.9 | 716.0 | +7.5% |

### Key Observations

1. **MPC benefits are strongest with moderate heterogeneity (2x-4x)**
   - 2x heterogeneity: 15.7% improvement
   - 4x heterogeneity: 16.2% improvement

2. **Homogeneous servers don't benefit from MPC**
   - -8.8% (slight overhead from MPC computation)

3. **Extreme heterogeneity (6x) shows diminishing returns**
   - 7.5% improvement
   - Slow servers become too slow to be useful regardless of routing

### When to Use MPC

| Scenario | Recommendation |
|----------|---------------|
| Homogeneous servers | Disable MPC (no benefit) |
| 2x-4x capacity difference | Enable MPC (15-16% improvement) |
| 6x+ capacity difference | Enable MPC (7-8% improvement, consider removing slow servers) |

## Full Results

| Experiment | Mean (ms) | P50 (ms) | P99 (ms) | Std (ms) |
|------------|-----------|----------|----------|----------|
| het_2x_po2_mpc_steady | 444.2 | 287.9 | 2810.3 | 471.3 |
| het_2x_po2_steady | 526.8 | 332.2 | 2690.2 | 525.3 |
| het_2x_rr_mpc_steady | 548.9 | 358.5 | 2588.4 | 528.8 |
| het_4x_po2_mpc_steady | 551.9 | 359.7 | 3525.5 | 624.2 |
| het_4x_po2_steady | 658.9 | 333.7 | 5282.9 | 935.6 |
| het_4x_rr_mpc_steady | 958.9 | 475.6 | 6517.8 | 1203.4 |
| het_6x_po2_mpc_steady | 716.0 | 399.2 | 7139.1 | 1083.8 |
| het_6x_po2_steady | 773.9 | 318.4 | 7001.2 | 1374.1 |
| het_6x_rr_mpc_steady | 2235.1 | 372.9 | 25712.6 | 4782.0 |
| homogeneous_po2_mpc_steady | 848.2 | 592.3 | 2887.2 | 675.6 |
| homogeneous_po2_steady | 779.4 | 553.7 | 2902.6 | 692.7 |
| homogeneous_rr_mpc_steady | 930.6 | 657.0 | 3028.3 | 780.1 |
