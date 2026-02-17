
# combined_plots_with_weights.py
# Creates a single figure with multiple subplots:
# Row 1: Latency over time | Throughput (RPS)
# Row 2: Latency histogram (spans both columns)
# Row 3: Decode inflight per node | Service rate per node
# Row 4: Node weights per node | Service rate efficiency (rate / weight) per node

import ast
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def safe_eval(x):
    """Safely parse dict-like strings into Python dicts."""
    try:
        return ast.literal_eval(str(x))
    except Exception:
        return {}

# ---------- Load data ----------
requests = pd.read_csv("requests.csv")
metrics = pd.read_csv("metrics.csv")

# Normalize timestamps
t0 = min(requests["timestamp"].min(), metrics["timestamp"].min())
requests["t"] = requests["timestamp"] - t0
metrics["t"] = metrics["timestamp"] - t0

# ---------- Pre-compute data ----------
# Throughput (RPS): bucket requests into 1-second bins
rps = (
    requests
    .groupby(requests["t"].astype(int))
    .size()
    .rename("rps")
)

# Expand dict-like columns safely
decode_df = pd.json_normalize(metrics["decode_inflight"].apply(safe_eval))
decode_df["t"] = metrics["t"]

sr_df = pd.json_normalize(metrics["service_rate"].apply(safe_eval))
sr_df["t"] = metrics["t"]

weights_df = pd.json_normalize(metrics["node_weights"].apply(safe_eval))
weights_df["t"] = metrics["t"]

# Build a unified set of node keys across all metrics for consistent coloring
node_cols = sorted(set(
    [c for c in decode_df.columns if c != "t"] +
    [c for c in sr_df.columns if c != "t"] +
    [c for c in weights_df.columns if c != "t"]
))

# Consistent color mapping per node
palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_map = {node: palette[i % len(palette)] for i, node in enumerate(node_cols)}

# Compute service rate efficiency (rate / weight) per node
eff_df = pd.DataFrame({"t": metrics["t"]})
for node in node_cols:
    rate = sr_df[node] if node in sr_df.columns else pd.Series([float("nan")] * len(sr_df))
    weight = weights_df[node] if node in weights_df.columns else pd.Series([float("nan")] * len(weights_df))
    eff_df[node] = rate / weight

# If you prefer "weighted service rate" instead of efficiency, swap the line above with:
# eff_df[node] = rate * weight

# ---------- Build single figure with subplots ----------
plt.figure(figsize=(18, 12))
gs = GridSpec(nrows=4, ncols=2, hspace=0.4, wspace=0.28)

# Plot 1: Latency over time (scatter)
ax1 = plt.subplot(gs[0, 0])
ax1.scatter(requests["t"], requests["latency"], s=6, alpha=0.6)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Latency (s)")
ax1.set_title("Request Latency Over Time")

# Plot 3: Throughput (RPS)
ax3 = plt.subplot(gs[0, 1], sharex=ax1)
ax3.plot(rps.index, rps.values, color="#1f77b4")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Requests / second")
ax3.set_title("Throughput Over Time")

# Plot 2: Latency histogram (spans both columns)
ax2 = plt.subplot(gs[1, :])
ax2.hist(requests["latency"], bins=100, color="#ff7f0e", edgecolor="none")
ax2.set_xlabel("Latency (s)")
ax2.set_ylabel("Count")
ax2.set_title("Latency Distribution")

# Plot 4: Decode inflight (per node)
ax4 = plt.subplot(gs[2, 0], sharex=ax1)
for node in node_cols:
    if node in decode_df.columns:
        ax4.plot(decode_df["t"], decode_df[node], label=f"node {node}", color=color_map[node])
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Inflight Decodes")
ax4.set_title("Decode Inflight Over Time")
handles, labels = ax4.get_legend_handles_labels()
ax4.legend(fontsize=8, ncol=2, frameon=False, loc="upper left")

# Plot 5: Service rate per node
ax5 = plt.subplot(gs[2, 1], sharex=ax1)
for node in node_cols:
    if node in sr_df.columns:
        ax5.plot(sr_df["t"], sr_df[node], label=f"node {node}", color=color_map[node])
ax5.set_xlabel("Time (s)")
ax5.set_ylabel("Service Rate")
ax5.set_title("Service Rate Per Node")
ax5.legend(fontsize=8, ncol=2, frameon=False, loc="upper left")

# Plot 6: Node weights per node
ax6 = plt.subplot(gs[3, 0], sharex=ax1)
for node in node_cols:
    if node in weights_df.columns:
        ax6.plot(weights_df["t"], weights_df[node], label=f"node {node}", color=color_map[node])
ax6.set_xlabel("Time (s)")
ax6.set_ylabel("Node Weight")
ax6.set_title("Node Weights Over Time")
ax6.legend(fontsize=8, ncol=2, frameon=False, loc="upper left")

# Plot 7: Service rate efficiency (rate / weight)
ax7 = plt.subplot(gs[3, 1], sharex=ax1)
for node in node_cols:
    if node in eff_df.columns:
        ax7.plot(eff_df["t"], eff_df[node], label=f"node {node}", color=color_map[node])
ax7.set_xlabel("Time (s)")
ax7.set_ylabel("Service Rate / Weight")
ax7.set_title("Service Rate Efficiency Per Node")
ax7.legend(fontsize=8, ncol=2, frameon=False, loc="upper left")

# Global figure title
plt.suptitle("System Performance Overview (Weights Considered)", fontsize=16, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Show once
plt.show()
