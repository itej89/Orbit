#!/usr/bin/env python3
"""
Comprehensive Experiment Runner for Orbit

Tests all combinations of:
- Server heterogeneity levels (homogeneous, 2x, 4x, 6x difference)
- Routing policies (random, rr, po2)
- MPC enabled/disabled
- Workload patterns (steady, bursty)

Generates comparison tables and figures for the paper.
"""

import asyncio
import subprocess
import time
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import signal

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ServerConfig:
    """Configuration for a single server."""
    name: str
    port: int
    preset: str  # fast, medium, slow, variable
    extra_args: List[str] = None
    
    def __post_init__(self):
        if self.extra_args is None:
            self.extra_args = []


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    prefill_servers: List[ServerConfig]
    decode_servers: List[ServerConfig]
    routing_policy: str  # random, rr, po2
    mpc_enabled: bool
    workload_pattern: str  # steady, bursty
    total_requests: int = 300
    target_rps: float = 15.0
    concurrency: int = 24


# Server preset definitions (delay in ms)
SERVER_PRESETS = {
    "fast": {"prefill_base": 5, "decode_token": 8, "capacity": 16},
    "medium": {"prefill_base": 15, "decode_token": 15, "capacity": 8},
    "slow": {"prefill_base": 30, "decode_token": 30, "capacity": 4},
    "very_slow": {"prefill_base": 50, "decode_token": 50, "capacity": 2},
}

# =============================================================================
# Experiment Definitions
# =============================================================================

def create_experiments() -> List[ExperimentConfig]:
    """Create all experiment configurations."""
    experiments = []
    
    # Heterogeneity configurations
    heterogeneity_configs = {
        "homogeneous": {
            "prefill": [("P1", "medium"), ("P2", "medium")],
            "decode": [("D1", "medium"), ("D2", "medium")],
        },
        "het_2x": {
            "prefill": [("P1", "fast"), ("P2", "medium")],
            "decode": [("D1", "fast"), ("D2", "medium")],
        },
        "het_4x": {
            "prefill": [("P1", "fast"), ("P2", "slow")],
            "decode": [("D1", "fast"), ("D2", "slow")],
        },
        "het_6x": {
            "prefill": [("P1", "fast"), ("P2", "very_slow")],
            "decode": [("D1", "fast"), ("D2", "very_slow")],
        },
    }
    
    # Routing policies to test
    policies = ["random", "rr", "po2"]
    
    # MPC on/off
    mpc_options = [False, True]
    
    # Workload patterns
    patterns = ["steady", "bursty"]
    
    # Generate all combinations
    for het_name, het_config in heterogeneity_configs.items():
        for policy in policies:
            for mpc_enabled in mpc_options:
                for pattern in patterns:
                    # Skip MPC for random policy (no benefit)
                    if policy == "random" and mpc_enabled:
                        continue
                    
                    # Create server configs
                    prefill_servers = []
                    for i, (name, preset) in enumerate(het_config["prefill"]):
                        prefill_servers.append(ServerConfig(
                            name=name,
                            port=8100 + i,
                            preset=preset
                        ))
                    
                    decode_servers = []
                    for i, (name, preset) in enumerate(het_config["decode"]):
                        decode_servers.append(ServerConfig(
                            name=name,
                            port=8200 + i,
                            preset=preset
                        ))
                    
                    exp_name = f"{het_name}_{policy}"
                    if mpc_enabled:
                        exp_name += "_mpc"
                    exp_name += f"_{pattern}"
                    
                    experiments.append(ExperimentConfig(
                        name=exp_name,
                        prefill_servers=prefill_servers,
                        decode_servers=decode_servers,
                        routing_policy=policy,
                        mpc_enabled=mpc_enabled,
                        workload_pattern=pattern,
                    ))
    
    return experiments


# =============================================================================
# Process Management
# =============================================================================

class ProcessManager:
    """Manages background processes for servers and router."""
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
    
    def start_server(self, name: str, cmd: List[str], cwd: str = None) -> bool:
        """Start a server process."""
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.processes[name] = proc
            time.sleep(0.5)  # Give server time to start
            return proc.poll() is None
        except Exception as e:
            print(f"Failed to start {name}: {e}")
            return False
    
    def stop_all(self):
        """Stop all processes."""
        for name, proc in self.processes.items():
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                proc.kill()
        self.processes.clear()
    
    def is_running(self, name: str) -> bool:
        """Check if a process is still running."""
        if name not in self.processes:
            return False
        return self.processes[name].poll() is None


def get_script_dir() -> Path:
    """Get the directory containing this script."""
    return Path(__file__).parent.absolute()


# =============================================================================
# Experiment Runner
# =============================================================================

def run_experiment(
    exp: ExperimentConfig,
    script_dir: Path,
    results_dir: Path,
    pm: ProcessManager
) -> Dict:
    """Run a single experiment."""
    
    print(f"\n{'='*70}")
    print(f"Experiment: {exp.name}")
    print(f"{'='*70}")
    print(f"  Policy: {exp.routing_policy}, MPC: {exp.mpc_enabled}")
    print(f"  Pattern: {exp.workload_pattern}")
    print(f"  Prefill servers: {[(s.name, s.preset) for s in exp.prefill_servers]}")
    print(f"  Decode servers: {[(s.name, s.preset) for s in exp.decode_servers]}")
    
    # Start prefill servers
    for server in exp.prefill_servers:
        preset = SERVER_PRESETS[server.preset]
        cmd = [
            sys.executable,
            str(script_dir / "prefill_server_v2.py"),
            "--port", str(server.port),
            "--base-delay", str(preset["prefill_base"] / 1000),
            "--capacity", str(preset["capacity"]),
            "--variance", "0.3",
        ]
        if not pm.start_server(f"prefill_{server.name}", cmd, str(script_dir)):
            print(f"  ERROR: Failed to start prefill server {server.name}")
            return {"error": f"Failed to start {server.name}"}
        print(f"  Started prefill server {server.name} on port {server.port}")
    
    # Start decode servers
    for server in exp.decode_servers:
        preset = SERVER_PRESETS[server.preset]
        cmd = [
            sys.executable,
            str(script_dir / "decode_server_v2.py"),
            "--port", str(server.port),
            "--base-token-delay", str(preset["decode_token"] / 1000),
            "--capacity", str(preset["capacity"]),
            "--variance", "0.3",
        ]
        if not pm.start_server(f"decode_{server.name}", cmd, str(script_dir)):
            print(f"  ERROR: Failed to start decode server {server.name}")
            return {"error": f"Failed to start {server.name}"}
        print(f"  Started decode server {server.name} on port {server.port}")
    
    time.sleep(1)  # Wait for servers to initialize
    
    # Start router
    prefill_hosts = " ".join(["127.0.0.1"] * len(exp.prefill_servers))
    prefill_ports = " ".join([str(s.port) for s in exp.prefill_servers])
    decode_hosts = " ".join(["127.0.0.1"] * len(exp.decode_servers))
    decode_ports = " ".join([str(s.port) for s in exp.decode_servers])
    
    router_cmd = [
        sys.executable,
        str(script_dir / "router_v2.py"),
        "--prefiller-hosts", *prefill_hosts.split(),
        "--prefiller-ports", *prefill_ports.split(),
        "--decoder-hosts", *decode_hosts.split(),
        "--decoder-ports", *decode_ports.split(),
        "--policy", exp.routing_policy,
        "--port", "8000",
    ]
    if exp.mpc_enabled:
        router_cmd.append("--enable-mpc")
    
    if not pm.start_server("router", router_cmd, str(script_dir)):
        print("  ERROR: Failed to start router")
        return {"error": "Failed to start router"}
    print(f"  Started router on port 8000")
    
    time.sleep(2)  # Wait for router to initialize
    
    # Run benchmark
    exp_results_dir = results_dir / exp.name
    exp_results_dir.mkdir(parents=True, exist_ok=True)
    
    benchmark_cmd = [
        sys.executable,
        str(script_dir / "benchmark_v2.py"),
        "--requests", str(exp.total_requests),
        "--rps", str(exp.target_rps),
        "--concurrency", str(exp.concurrency),
        "--pattern", exp.workload_pattern,
        "--output-dir", str(exp_results_dir.parent),
        "--name", exp.name,
    ]
    
    print(f"  Running benchmark...")
    try:
        result = subprocess.run(
            benchmark_cmd,
            cwd=str(script_dir),
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        if result.returncode != 0:
            print(f"  Benchmark failed: {result.stderr[:500]}")
    except subprocess.TimeoutExpired:
        print("  Benchmark timed out")
    
    # Load results
    summary_file = exp_results_dir / "summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        print(f"  Results: mean={summary.get('latency', {}).get('mean_ms', 'N/A'):.1f}ms, "
              f"p99={summary.get('latency', {}).get('p99_ms', 'N/A'):.1f}ms")
    else:
        summary = {"error": "No results"}
    
    # Stop all processes
    pm.stop_all()
    time.sleep(1)
    
    return summary


def generate_comparison_table(results_dir: Path) -> None:
    """Generate comparison table from all experiment results."""
    
    print(f"\n{'='*70}")
    print("RESULTS COMPARISON")
    print(f"{'='*70}")
    
    results = {}
    for exp_dir in sorted(results_dir.iterdir()):
        summary_file = exp_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                results[exp_dir.name] = json.load(f)
    
    # Print table header
    print(f"\n{'Experiment':<40} {'Mean(ms)':<10} {'P99(ms)':<10} {'Std(ms)':<10} {'Success%':<10}")
    print("-" * 80)
    
    for name, data in sorted(results.items()):
        if "error" in data:
            print(f"{name:<40} ERROR")
            continue
        
        lat = data.get("latency", {})
        print(f"{name:<40} "
              f"{lat.get('mean_ms', 0):>8.1f}  "
              f"{lat.get('p99_ms', 0):>8.1f}  "
              f"{lat.get('std_ms', 0):>8.1f}  "
              f"{data.get('success_rate', 0):>8.1f}")
    
    # Generate improvement analysis
    print(f"\n{'='*70}")
    print("MPC IMPROVEMENT ANALYSIS")
    print(f"{'='*70}")
    
    # Find baseline vs MPC pairs
    baselines = {}
    mpc_results = {}
    
    for name, data in results.items():
        if "error" in data:
            continue
        
        if "_mpc_" in name:
            base_name = name.replace("_mpc_", "_")
            mpc_results[base_name] = data
        else:
            baselines[name] = data
    
    print(f"\n{'Configuration':<35} {'Baseline(ms)':<15} {'MPC(ms)':<15} {'Improvement':<15}")
    print("-" * 80)
    
    for base_name, base_data in sorted(baselines.items()):
        if base_name in mpc_results:
            mpc_data = mpc_results[base_name]
            base_lat = base_data.get("latency", {}).get("mean_ms", 0)
            mpc_lat = mpc_data.get("latency", {}).get("mean_ms", 0)
            
            if base_lat > 0:
                improvement = (base_lat - mpc_lat) / base_lat * 100
                print(f"{base_name:<35} {base_lat:>12.1f}   {mpc_lat:>12.1f}   {improvement:>+10.1f}%")
    
    # Save comprehensive results
    with open(results_dir / "all_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAll results saved to: {results_dir / 'all_results.json'}")


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run all Orbit experiments")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick test with fewer experiments")
    parser.add_argument("--output-dir", default="results",
                        help="Output directory for results")
    parser.add_argument("--filter", type=str, default=None,
                        help="Only run experiments matching this pattern")
    args = parser.parse_args()
    
    script_dir = get_script_dir()
    results_dir = script_dir / args.output_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create experiments
    experiments = create_experiments()
    
    # Filter if requested
    if args.filter:
        experiments = [e for e in experiments if args.filter in e.name]
    
    # Quick mode: only test key scenarios
    if args.quick:
        quick_names = [
            "homogeneous_po2_steady",
            "homogeneous_po2_mpc_steady",
            "het_4x_po2_steady",
            "het_4x_po2_mpc_steady",
        ]
        experiments = [e for e in experiments if e.name in quick_names]
    
    print(f"Running {len(experiments)} experiments")
    print(f"Results will be saved to: {results_dir}")
    
    # Process manager
    pm = ProcessManager()
    
    # Signal handler for cleanup
    def cleanup(sig, frame):
        print("\nCleaning up...")
        pm.stop_all()
        sys.exit(1)
    
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    # Run experiments
    all_results = {}
    for i, exp in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] ", end="")
        try:
            result = run_experiment(exp, script_dir, results_dir, pm)
            all_results[exp.name] = result
        except Exception as e:
            print(f"Experiment failed: {e}")
            all_results[exp.name] = {"error": str(e)}
            pm.stop_all()
    
    # Generate comparison
    generate_comparison_table(results_dir)
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()
