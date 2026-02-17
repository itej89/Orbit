#!/usr/bin/env python3
"""
Analyze Orbit experiment results and generate paper figures.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter

# Plot styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.figsize': (6, 4),
    'figure.dpi': 150,
})

COLORS = {
    'mpc_po2': '#2563EB',      # Blue
    'po2': '#DC2626',           # Red
    'mpc_rr': '#059669',        # Green
    'rr': '#F59E0B',            # Amber
    'random': '#6B7280',        # Gray
    'cache_aware': '#8B5CF6',   # Purple
}

LABELS = {
    'mpc_po2': 'Orbit (MPC-PO2)',
    'po2': 'Vanilla PO2',
    'mpc_rr': 'Orbit (MPC-RR)',
    'rr': 'Round Robin',
    'random': 'Random',
    'cache_aware': 'Cache-Aware',
}


def load_experiment(exp_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load experiment data from directory."""
    requests = pd.read_csv(exp_dir / 'requests.csv')
    metrics = pd.read_csv(exp_dir / 'metrics.csv')
    
    summary_path = exp_dir / 'summary.json'
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = compute_summary(requests)
    
    return requests, metrics, summary


def compute_summary(requests: pd.DataFrame) -> dict:
    """Compute latency statistics from requests."""
    latency_ms = requests['latency'] * 1000
    
    return {
        'mean_ms': latency_ms.mean(),
        'p50_ms': latency_ms.quantile(0.5),
        'p90_ms': latency_ms.quantile(0.9),
        'p95_ms': latency_ms.quantile(0.95),
        'p99_ms': latency_ms.quantile(0.99),
        'std_ms': latency_ms.std(),
        'total_requests': len(requests),
        'success_rate': (requests['status'] == 200).mean() * 100,
    }


def load_all_experiments(results_dir: Path) -> Dict[str, dict]:
    """Load all experiments from results directory."""
    experiments = {}
    
    for exp_dir in sorted(results_dir.iterdir()):
        if exp_dir.is_dir() and (exp_dir / 'requests.csv').exists():
            requests, metrics, summary = load_experiment(exp_dir)
            experiments[exp_dir.name] = {
                'requests': requests,
                'metrics': metrics,
                'summary': summary,
            }
    
    return experiments


def plot_latency_comparison(experiments: Dict[str, dict], output_path: Path):
    """Generate latency comparison bar chart."""
    # Group by routing policy
    policies = ['rr', 'po2', 'mpc_rr', 'mpc_po2']
    
    # Find experiments for each policy
    data = {p: [] for p in policies}
    for exp_name, exp_data in experiments.items():
        for policy in policies:
            if policy in exp_name:
                data[policy].append(exp_data['summary']['mean_ms'])
                break
    
    # Compute averages
    means = [np.mean(data[p]) if data[p] else 0 for p in policies]
    stds = [np.std(data[p]) if len(data[p]) > 1 else 0 for p in policies]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    x = np.arange(len(policies))
    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color=[COLORS[p] for p in policies],
                  edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Mean Latency (ms)')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[p] for p in policies], rotation=15, ha='right')
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, mean in zip(bars, means):
        if mean > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f'{mean:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_latency_cdf(experiments: Dict[str, dict], output_path: Path):
    """Generate latency CDF comparison."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    for exp_name, exp_data in experiments.items():
        # Determine policy from name
        policy = None
        for p in COLORS.keys():
            if p in exp_name:
                policy = p
                break
        
        if policy is None:
            continue
        
        latency = exp_data['requests']['latency'] * 1000
        sorted_latency = np.sort(latency)
        cdf = np.arange(1, len(sorted_latency) + 1) / len(sorted_latency)
        
        ax.plot(sorted_latency, cdf, 
                color=COLORS[policy], 
                label=f"{LABELS[policy]} ({exp_name.split('_')[0]})",
                alpha=0.7)
    
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('CDF')
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=8)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_throughput_over_time(experiments: Dict[str, dict], output_path: Path):
    """Generate throughput over time comparison."""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    for exp_name, exp_data in experiments.items():
        # Determine policy
        policy = None
        for p in COLORS.keys():
            if p in exp_name:
                policy = p
                break
        
        if policy is None:
            continue
        
        requests = exp_data['requests'].copy()
        requests['timestamp'] = pd.to_datetime(requests['timestamp'], unit='s')
        requests.set_index('timestamp', inplace=True)
        
        # Resample to 1-second bins
        throughput = requests.resample('1s').size()
        
        # Relative time
        time_sec = (throughput.index - throughput.index[0]).total_seconds()
        
        ax.plot(time_sec, throughput.values,
                color=COLORS[policy],
                label=LABELS[policy],
                alpha=0.8)
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Throughput (requests/s)')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_weight_trajectories(experiments: Dict[str, dict], output_path: Path):
    """Plot MPC weight trajectories over time."""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    for exp_name, exp_data in experiments.items():
        if 'mpc' not in exp_name:
            continue
        
        metrics = exp_data['metrics'].copy()
        if 'node_weights' not in metrics.columns:
            continue
        
        # Parse weights
        try:
            metrics['weights_parsed'] = metrics['node_weights'].apply(
                lambda x: eval(x) if isinstance(x, str) else x
            )
        except:
            continue
        
        # Plot each worker's weight
        for i in range(2):  # Assume 2 workers
            weights = metrics['weights_parsed'].apply(
                lambda x: x.get(i, 1.0) if isinstance(x, dict) else 1.0
            )
            time_sec = metrics['timestamp'] - metrics['timestamp'].iloc[0]
            
            ax.plot(time_sec, weights,
                    label=f'{exp_name} - Worker {i}',
                    alpha=0.7)
    
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('MPC Weight')
    ax.set_ylim(0, 3.5)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_latex_table(experiments: Dict[str, dict], output_path: Path):
    """Generate LaTeX table for paper."""
    rows = []
    
    for exp_name, exp_data in sorted(experiments.items()):
        summary = exp_data['summary']
        rows.append({
            'Experiment': exp_name,
            'Mean (ms)': f"{summary['mean_ms']:.1f}",
            'P50 (ms)': f"{summary['p50_ms']:.1f}",
            'P99 (ms)': f"{summary['p99_ms']:.1f}",
            'Std (ms)': f"{summary['std_ms']:.1f}",
            'Requests': summary['total_requests'],
        })
    
    df = pd.DataFrame(rows)
    
    latex = df.to_latex(index=False, escape=False)
    
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"Saved: {output_path}")
    print(df.to_string())


def main():
    parser = argparse.ArgumentParser(description='Analyze Orbit experiment results')
    parser.add_argument('results_dir', type=Path, help='Directory containing experiment results')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for figures')
    args = parser.parse_args()
    
    output_dir = args.output_dir or args.results_dir / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading experiments from: {args.results_dir}")
    experiments = load_all_experiments(args.results_dir)
    print(f"Found {len(experiments)} experiments")
    
    if not experiments:
        print("No experiments found!")
        return
    
    # Generate all plots
    print("\nGenerating figures...")
    plot_latency_comparison(experiments, output_dir / 'latency_comparison.pdf')
    plot_latency_cdf(experiments, output_dir / 'latency_cdf.pdf')
    plot_throughput_over_time(experiments, output_dir / 'throughput_time.pdf')
    plot_weight_trajectories(experiments, output_dir / 'mpc_weights.pdf')
    generate_latex_table(experiments, output_dir / 'results_table.tex')
    
    print(f"\nAll figures saved to: {output_dir}")


if __name__ == '__main__':
    main()
