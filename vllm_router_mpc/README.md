# vLLM Router MPC Integration

This directory contains the MPC (Model Predictive Control) integration for the vLLM router.

## Overview

Orbit's MPC algorithm is integrated as an optional enhancement layer for the existing vLLM router policies. It can be enabled/disabled via configuration.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      vLLM Router                             │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ round_robin  │    │ power_of_two │    │ cache_aware  │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                   │                   │           │
│         └─────────┬─────────┴─────────┬─────────┘           │
│                   │                   │                      │
│         ┌─────────▼─────────┐         │                      │
│         │   MPC Controller  │◄────────┘ (optional)           │
│         │   (Orbit)         │                                │
│         └─────────┬─────────┘                                │
│                   │                                          │
│         ┌─────────▼─────────┐                                │
│         │  Weighted Router  │                                │
│         └───────────────────┘                                │
└─────────────────────────────────────────────────────────────┘
```

## Files to Add to vLLM Router

```
vllm-router/
├── src/
│   ├── mpc/                    # NEW: MPC module
│   │   ├── mod.rs              # Module definition
│   │   ├── controller.rs       # MPC controller implementation
│   │   ├── estimator.rs        # Rate/service estimation
│   │   └── solver.rs           # QP solver interface
│   ├── routing/
│   │   ├── mod.rs
│   │   ├── round_robin.rs
│   │   ├── power_of_two.rs
│   │   ├── consistent_hash.rs
│   │   ├── cache_aware.rs
│   │   └── mpc_weighted.rs     # NEW: MPC-weighted wrapper
│   └── lib.rs                  # Add mpc module
├── py_src/
│   └── vllm_router/
│       └── mpc.py              # Python MPC controller (alternative)
└── Cargo.toml                  # Add osqp dependency
```

## Configuration Options

### CLI Arguments

```bash
vllm-router \
    --worker-urls http://worker1:8000 http://worker2:8000 \
    --policy power_of_two \
    --enable-mpc \                    # Enable MPC layer
    --mpc-horizon 5 \                 # Prediction horizon
    --mpc-timestep 0.1 \              # Controller timestep (seconds)
    --mpc-regularization 0.5 \        # Weight regularization
    --mpc-weight-bounds 0.1,3.0       # Min,max weight bounds
```

### Environment Variables

```bash
VLLM_ROUTER_MPC_ENABLED=true
VLLM_ROUTER_MPC_HORIZON=5
VLLM_ROUTER_MPC_TIMESTEP=0.1
VLLM_ROUTER_MPC_LAMBDA=0.5
```

## Implementation Plan

### Phase 1: Python Prototype (Quick Validation)

Use the existing Orbit Python implementation as a sidecar process.

```
┌─────────────┐     gRPC/HTTP      ┌─────────────┐
│ vLLM Router │◄──────────────────►│ MPC Sidecar │
│   (Rust)    │   weight updates   │  (Python)   │
└─────────────┘                    └─────────────┘
```

**Pros:** Fast to implement, reuses existing code
**Cons:** Additional latency, extra process

### Phase 2: Rust Native Implementation

Port MPC to Rust for minimal overhead.

**Dependencies:**
- `osqp` - Rust bindings for OSQP solver
- `ndarray` - Matrix operations

### Phase 3: Integration with Prefill-Decode

Extend MPC to handle separate prefill/decode weight optimization.

## MPC Algorithm (Rust Pseudocode)

```rust
struct MpcController {
    horizon: usize,
    timestep: f64,
    target_q: f64,
    lambda: f64,
    weight_bounds: (f64, f64),
    
    // State
    weights: HashMap<WorkerId, f64>,
    service_rates: HashMap<WorkerId, RateEstimator>,
    arrival_rate: RateEstimator,
}

impl MpcController {
    async fn control_loop(&mut self, metrics: &Metrics) {
        loop {
            // Get current state
            let queues = metrics.get_queue_depths();
            let arrival = self.arrival_rate.get();
            
            // Solve MPC for each worker
            for (worker_id, q0) in queues.iter() {
                let mu = self.service_rates.get(worker_id).get();
                let w_new = self.solve_qp(q0, arrival, mu);
                
                // Smooth weight update
                let w_old = self.weights.get(worker_id).unwrap_or(1.0);
                let w_smooth = 0.25 * w_new + 0.75 * w_old;
                self.weights.insert(worker_id, w_smooth.clamp(0.1, 3.0));
            }
            
            tokio::time::sleep(Duration::from_secs_f64(self.timestep)).await;
        }
    }
    
    fn solve_qp(&self, q0: f64, arrival: f64, mu: f64) -> f64 {
        // Formulate and solve QP using OSQP
        // min Σ (q[k+1] - q*)² + λ(w[k] - 1)²
        // s.t. q[k+1] = q[k] + dt*(arrival*p - mu)
        //      w_min <= w <= w_max
        //      q >= 0
        
        // ... OSQP solver call ...
    }
}
```

## Weighted Routing Integration

```rust
// In power_of_two.rs (modified)
impl PowerOfTwo {
    fn select(&self, workers: &[Worker], mpc_weights: Option<&HashMap<WorkerId, f64>>) -> Worker {
        let (a, b) = self.sample_two(workers);
        
        let score_a = match mpc_weights {
            Some(w) => a.queue_depth / w.get(&a.id).unwrap_or(&1.0),
            None => a.queue_depth as f64,
        };
        let score_b = match mpc_weights {
            Some(w) => b.queue_depth / w.get(&b.id).unwrap_or(&1.0),
            None => b.queue_depth as f64,
        };
        
        if score_a <= score_b { a } else { b }
    }
}
```

## Testing

### Unit Tests

```bash
cargo test mpc
```

### Integration Tests

```bash
# Start test servers
./scripts/start_test_servers.sh

# Run router with MPC
cargo run -- --policy power_of_two --enable-mpc

# Run benchmark
python benchmark.py
```

### Comparison Tests

```bash
# Without MPC
./run_benchmark.sh --policy power_of_two --output results_po2.csv

# With MPC
./run_benchmark.sh --policy power_of_two --enable-mpc --output results_mpc_po2.csv

# Compare
python compare_results.py results_po2.csv results_mpc_po2.csv
```

## Metrics Exposed

The MPC controller exposes additional Prometheus metrics:

```
# MPC weight for each worker
vllm_router_mpc_weight{worker_id="0"} 1.2
vllm_router_mpc_weight{worker_id="1"} 0.8

# MPC solver statistics
vllm_router_mpc_solve_time_seconds 0.0005
vllm_router_mpc_solver_iterations 3

# Estimated rates
vllm_router_mpc_arrival_rate 45.2
vllm_router_mpc_service_rate{worker_id="0"} 12.5
vllm_router_mpc_service_rate{worker_id="1"} 8.3
```

## Fork and Development

### Clone vLLM Router

```bash
# Fork https://github.com/vllm-project/router to your account
# Then clone
git clone https://github.com/YOUR_USERNAME/router.git vllm-router-mpc
cd vllm-router-mpc
git checkout -b feature/mpc-integration
```

### Add MPC Module

```bash
# Create MPC module structure
mkdir -p src/mpc
touch src/mpc/mod.rs
touch src/mpc/controller.rs
touch src/mpc/estimator.rs
```

### Build and Test

```bash
cargo build --release
cargo test
```

## References

- [vLLM Router GitHub](https://github.com/vllm-project/router)
- [OSQP Solver](https://osqp.org/)
- [Orbit MPC Implementation](../src/orbit/router.py)
