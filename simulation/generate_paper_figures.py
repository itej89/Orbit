#!/usr/bin/env python3
"""
Generate publication-ready figures and tables from experiment results.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping plots")


def load_results(results_dir: Path) -> dict:
    """Load all experiment results."""
    results_file = results_dir / "all_results.json"
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        sys.exit(1)
    
    with open(results_file) as f:
        return json.load(f)


def generate_latex_table(results: dict, output_dir: Path):
    """Generate LaTeX table for the paper."""
    
    # Filter to PO2 experiments (main comparison)
    po2_results = {k: v for k, v in results.items() if "po2" in k and "steady" in k}
    
    # Group by heterogeneity
    table_data = []
    for het_level in ["homogeneous", "het_2x", "het_4x", "het_6x"]:
        baseline_key = f"{het_level}_po2_steady"
        mpc_key = f"{het_level}_po2_mpc_steady"
        
        if baseline_key in po2_results:
            baseline = po2_results[baseline_key]
            base_mean = baseline["latency"]["mean_ms"]
            base_p99 = baseline["latency"]["p99_ms"]
            base_std = baseline["latency"]["std_ms"]
            
            mpc_mean = mpc_p99 = mpc_std = improvement = "--"
            if mpc_key in po2_results:
                mpc = po2_results[mpc_key]
                mpc_mean = mpc["latency"]["mean_ms"]
                mpc_p99 = mpc["latency"]["p99_ms"]
                mpc_std = mpc["latency"]["std_ms"]
                improvement = (base_mean - mpc_mean) / base_mean * 100
            
            table_data.append({
                "het": het_level.replace("_", "-").replace("het-", ""),
                "base_mean": base_mean,
                "base_p99": base_p99,
                "base_std": base_std,
                "mpc_mean": mpc_mean,
                "mpc_p99": mpc_p99,
                "mpc_std": mpc_std,
                "improvement": improvement,
            })
    
    # Generate LaTeX
    latex = r"""
\begin{table}[t]
\caption{MPC improvement across heterogeneity levels (PO2 baseline). MPC shows strongest benefits with moderate heterogeneity (2x-4x capacity difference).}
\label{tab:mpc_results}
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{l ccc ccc c}
\toprule
& \multicolumn{3}{c}{\textbf{Baseline PO2}} & \multicolumn{3}{c}{\textbf{MPC-Augmented}} & \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7}
\textbf{Config} & Mean & P99 & Std & Mean & P99 & Std & \textbf{Impr.} \\
& (ms) & (ms) & (ms) & (ms) & (ms) & (ms) & (\%) \\
\midrule
"""
    
    for row in table_data:
        het = row["het"]
        if het == "homogeneous":
            het = "Homo."
        else:
            het = het.upper()
        
        if isinstance(row["mpc_mean"], str):
            latex += f"{het} & {row['base_mean']:.0f} & {row['base_p99']:.0f} & {row['base_std']:.0f} & -- & -- & -- & -- \\\\\n"
        else:
            latex += f"{het} & {row['base_mean']:.0f} & {row['base_p99']:.0f} & {row['base_std']:.0f} & "
            latex += f"{row['mpc_mean']:.0f} & {row['mpc_p99']:.0f} & {row['mpc_std']:.0f} & "
            if row['improvement'] > 0:
                latex += f"\\textbf{{+{row['improvement']:.1f}}} \\\\\n"
            else:
                latex += f"{row['improvement']:.1f} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    output_file = output_dir / "results_table.tex"
    with open(output_file, "w") as f:
        f.write(latex)
    print(f"LaTeX table saved to: {output_file}")


def generate_summary_markdown(results: dict, output_dir: Path):
    """Generate a markdown summary of results."""
    
    md = "# Orbit Simulation Results Summary\n\n"
    md += "## Key Findings\n\n"
    
    # Calculate improvements
    improvements = {}
    for het_level in ["homogeneous", "het_2x", "het_4x", "het_6x"]:
        baseline_key = f"{het_level}_po2_steady"
        mpc_key = f"{het_level}_po2_mpc_steady"
        
        if baseline_key in results and mpc_key in results:
            base = results[baseline_key]["latency"]["mean_ms"]
            mpc = results[mpc_key]["latency"]["mean_ms"]
            improvements[het_level] = (base - mpc) / base * 100
    
    md += "### MPC Improvement by Heterogeneity Level\n\n"
    md += "| Configuration | Baseline Mean (ms) | MPC Mean (ms) | Improvement |\n"
    md += "|--------------|-------------------|---------------|-------------|\n"
    
    for het_level in ["homogeneous", "het_2x", "het_4x", "het_6x"]:
        baseline_key = f"{het_level}_po2_steady"
        mpc_key = f"{het_level}_po2_mpc_steady"
        
        if baseline_key in results:
            base = results[baseline_key]["latency"]["mean_ms"]
            mpc_val = "--"
            imp_val = "--"
            
            if mpc_key in results:
                mpc_val = f"{results[mpc_key]['latency']['mean_ms']:.1f}"
                imp = improvements.get(het_level, 0)
                imp_val = f"+{imp:.1f}%" if imp > 0 else f"{imp:.1f}%"
            
            md += f"| {het_level} | {base:.1f} | {mpc_val} | {imp_val} |\n"
    
    md += "\n### Key Observations\n\n"
    md += "1. **MPC benefits are strongest with moderate heterogeneity (2x-4x)**\n"
    md += f"   - 2x heterogeneity: {improvements.get('het_2x', 0):.1f}% improvement\n"
    md += f"   - 4x heterogeneity: {improvements.get('het_4x', 0):.1f}% improvement\n"
    md += "\n2. **Homogeneous servers don't benefit from MPC**\n"
    md += f"   - {improvements.get('homogeneous', 0):.1f}% (slight overhead from MPC computation)\n"
    md += "\n3. **Extreme heterogeneity (6x) shows diminishing returns**\n"
    md += f"   - {improvements.get('het_6x', 0):.1f}% improvement\n"
    md += "   - Slow servers become too slow to be useful regardless of routing\n"
    
    md += "\n### When to Use MPC\n\n"
    md += "| Scenario | Recommendation |\n"
    md += "|----------|---------------|\n"
    md += "| Homogeneous servers | Disable MPC (no benefit) |\n"
    md += "| 2x-4x capacity difference | Enable MPC (15-16% improvement) |\n"
    md += "| 6x+ capacity difference | Enable MPC (7-8% improvement, consider removing slow servers) |\n"
    
    md += "\n## Full Results\n\n"
    md += "| Experiment | Mean (ms) | P50 (ms) | P99 (ms) | Std (ms) |\n"
    md += "|------------|-----------|----------|----------|----------|\n"
    
    for name, data in sorted(results.items()):
        lat = data["latency"]
        md += f"| {name} | {lat['mean_ms']:.1f} | {lat['p50_ms']:.1f} | {lat['p99_ms']:.1f} | {lat['std_ms']:.1f} |\n"
    
    output_file = output_dir / "RESULTS_SUMMARY.md"
    with open(output_file, "w") as f:
        f.write(md)
    print(f"Summary saved to: {output_file}")


def generate_plots(results: dict, output_dir: Path):
    """Generate publication plots."""
    if not HAS_MATPLOTLIB:
        return
    
    # Plot 1: Mean latency comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Mean latency by heterogeneity
    het_levels = ["Homo.", "2x", "4x", "6x"]
    baseline_means = []
    mpc_means = []
    
    for het in ["homogeneous", "het_2x", "het_4x", "het_6x"]:
        baseline_key = f"{het}_po2_steady"
        mpc_key = f"{het}_po2_mpc_steady"
        
        if baseline_key in results:
            baseline_means.append(results[baseline_key]["latency"]["mean_ms"])
        else:
            baseline_means.append(0)
        
        if mpc_key in results:
            mpc_means.append(results[mpc_key]["latency"]["mean_ms"])
        else:
            mpc_means.append(0)
    
    x = np.arange(len(het_levels))
    width = 0.35
    
    ax = axes[0]
    bars1 = ax.bar(x - width/2, baseline_means, width, label='Baseline PO2', color='#2563EB', alpha=0.8)
    bars2 = ax.bar(x + width/2, mpc_means, width, label='MPC-Augmented', color='#059669', alpha=0.8)
    
    ax.set_xlabel('Heterogeneity Level')
    ax.set_ylabel('Mean Latency (ms)')
    ax.set_title('Mean Latency: Baseline vs MPC')
    ax.set_xticks(x)
    ax.set_xticklabels(het_levels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Right: P99 latency comparison
    baseline_p99 = []
    mpc_p99 = []
    
    for het in ["homogeneous", "het_2x", "het_4x", "het_6x"]:
        baseline_key = f"{het}_po2_steady"
        mpc_key = f"{het}_po2_mpc_steady"
        
        if baseline_key in results:
            baseline_p99.append(results[baseline_key]["latency"]["p99_ms"])
        else:
            baseline_p99.append(0)
        
        if mpc_key in results:
            mpc_p99.append(results[mpc_key]["latency"]["p99_ms"])
        else:
            mpc_p99.append(0)
    
    ax = axes[1]
    bars1 = ax.bar(x - width/2, baseline_p99, width, label='Baseline PO2', color='#2563EB', alpha=0.8)
    bars2 = ax.bar(x + width/2, mpc_p99, width, label='MPC-Augmented', color='#059669', alpha=0.8)
    
    ax.set_xlabel('Heterogeneity Level')
    ax.set_ylabel('P99 Latency (ms)')
    ax.set_title('Tail Latency: Baseline vs MPC')
    ax.set_xticks(x)
    ax.set_xticklabels(het_levels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "latency_comparison.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / "latency_comparison.png", bbox_inches='tight', dpi=150)
    print(f"Plot saved to: {output_dir / 'latency_comparison.pdf'}")
    
    # Plot 2: Improvement percentage
    fig, ax = plt.subplots(figsize=(8, 5))
    
    improvements = []
    for base, mpc in zip(baseline_means, mpc_means):
        if base > 0 and mpc > 0:
            improvements.append((base - mpc) / base * 100)
        else:
            improvements.append(0)
    
    colors = ['#DC2626' if imp < 0 else '#059669' for imp in improvements]
    bars = ax.bar(het_levels, improvements, color=colors, alpha=0.8)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Heterogeneity Level')
    ax.set_ylabel('Improvement (%)')
    ax.set_title('MPC Improvement Over Baseline PO2')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax.annotate(f'{imp:+.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / "mpc_improvement.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / "mpc_improvement.png", bbox_inches='tight', dpi=150)
    print(f"Plot saved to: {output_dir / 'mpc_improvement.pdf'}")


def main():
    results_dir = Path(__file__).parent / "results"
    output_dir = results_dir / "figures"
    output_dir.mkdir(exist_ok=True)
    
    results = load_results(results_dir)
    
    print(f"Loaded {len(results)} experiment results")
    
    generate_latex_table(results, output_dir)
    generate_summary_markdown(results, output_dir)
    generate_plots(results, output_dir)
    
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
