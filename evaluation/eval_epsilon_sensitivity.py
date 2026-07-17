"""
epsilon_sweep_full.py
Epsilon sensitivity analysis and d vs d² comparison for POROS.
Run from the project root: python epsilon_sweep_full.py

Outputs
-------
  results/epsilon_sensitivity.png   two-panel: disconnected ratio and mean CBTD/step vs epsilon
  results/d_vs_d2.png               grouped bar: single-hop vs multi-hop under d and d²
"""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from config import DATASET_FOLDER
from graph import Graph, create_nodes
from node import Color
from query import _dijkstra_all

PRINCIPLED_EPS = 10.95
EPSILON_RANGE  = [3, 6, 9, 10.95, 12, 15, 18, 21]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _path_length(prev, target_id):
    """Count number of hops from source to target using prev map."""
    hops, cur = 0, target_id
    while prev.get(cur) is not None:
        hops += 1
        cur = prev[cur]
    return hops


def _rebuild_edges(graph, epsilon, use_d2=True):
    """Rebuild graph edges at given epsilon with d² or d weights."""
    graph.edges = defaultdict(list)
    for node_a in graph.nodes:
        for node_b in graph.nodes:
            if node_a.node_id == node_b.node_id:
                continue
            if node_b.tir <= node_a.tir:
                continue
            d = graph.metric.pairwise[(node_a.node_id, node_b.node_id)]
            if d <= epsilon:
                w = d ** 2 if use_d2 else d
                graph.edges[node_a.node_id].append((node_b.node_id, w))


# ── Epsilon sweep ─────────────────────────────────────────────────────────────

def sweep_epsilon(graph, epsilon_values):
    """
    For each epsilon value, rebuild edges and compute:
      disconnected   : % of RED source nodes with no reachable BLUE target
      mean_cbtd_step : mean CBTD per step across all paths from RED to reachable BLUE
    """
    red_nodes = [n for n in graph.nodes if n.color == Color.RED]
    blue_ids  = {n.node_id for n in graph.nodes if n.color == Color.BLUE}

    disconnected    = []
    mean_cbtd_steps = []

    for eps in epsilon_values:
        _rebuild_edges(graph, eps, use_d2=True)

        reachable_count = 0
        all_cbtd_steps  = []

        for source in red_nodes:
            dist, prev = _dijkstra_all(graph, source.node_id)
            reachable_blues = [nid for nid in dist if nid in blue_ids]

            if not reachable_blues:
                continue

            reachable_count += 1

            for target_id in reachable_blues:
                path, cur = [], target_id
                while cur is not None:
                    path.append(graph.node_index[cur])
                    cur = prev.get(cur)
                path.reverse()

                for j in range(len(path) - 1):
                    d = graph.metric.pairwise[(path[j].node_id, path[j + 1].node_id)]
                    all_cbtd_steps.append(d)

        pct  = (1 - reachable_count / len(red_nodes)) * 100
        mean = float(np.mean(all_cbtd_steps)) if all_cbtd_steps else 0.0
        disconnected.append(pct)
        mean_cbtd_steps.append(mean)
        print(f"  ε = {eps:5.2f}  |  disconnected = {pct:.1f}%  |  mean CBTD/step = {mean:.2f}")

    # restore principled epsilon with d² weights
    _rebuild_edges(graph, PRINCIPLED_EPS, use_d2=True)
    return disconnected, mean_cbtd_steps


# ── d vs d² comparison ────────────────────────────────────────────────────────

def compute_d_vs_d2(graph):
    """
    At the principled epsilon, compare d (linear) vs d² (quadratic) edge weights.
    For each RED source node, finds paths to all reachable BLUE targets and counts
    single-hop vs multi-hop paths under each weighting scheme.
    """
    red_nodes = [n for n in graph.nodes if n.color == Color.RED]
    blue_ids  = {n.node_id for n in graph.nodes if n.color == Color.BLUE}

    results = {}

    for label, use_d2 in [('d', False), ('d²', True)]:
        _rebuild_edges(graph, PRINCIPLED_EPS, use_d2=use_d2)

        single_hop = 0
        multi_hop  = 0

        for source in red_nodes:
            dist, prev = _dijkstra_all(graph, source.node_id)
            reachable_blues = [nid for nid in dist if nid in blue_ids]

            for target_id in reachable_blues:
                hops = _path_length(prev, target_id)
                if hops == 1:
                    single_hop += 1
                else:
                    multi_hop += 1

        results[label] = {'single': single_hop, 'multi': multi_hop}
        print(f"  {label:<6}  single-hop = {single_hop}  |  multi-hop = {multi_hop}")

    # restore d² weights
    _rebuild_edges(graph, PRINCIPLED_EPS, use_d2=True)
    return results


# ── Figures ───────────────────────────────────────────────────────────────────

TICK_SHOW = {3, 6, 9, 12, 15, 18, 21}


def _style(ax):
    ax.set_facecolor("whitesmoke")
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.2, color="gray")
    ax.tick_params(axis='both', labelsize=14)


def plot_epsilon_sensitivity(epsilon_values, disconnected, mean_cbtd_steps, save_path):
    pidx       = epsilon_values.index(PRINCIPLED_EPS)
    xtick_pos  = [v for v in epsilon_values if v in TICK_SHOW]
    xtick_labs = [str(v) for v in xtick_pos]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.subplots_adjust(wspace=0.32)

    for ax in (ax1, ax2):
        _style(ax)
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_labs, fontsize=15)
        ax.set_xlim(2, 22)
        ax.axvline(x=PRINCIPLED_EPS, color="tab:red", linewidth=1.5,
                   linestyle="--", zorder=2)

    # Left: Disconnected ratio
    ax1.plot(epsilon_values, disconnected, color="tab:blue", linewidth=2,
             marker='o', markersize=5, markerfacecolor="tab:blue", zorder=3)
    ax1.plot(PRINCIPLED_EPS, disconnected[pidx], 'o', color="tab:red",
             markersize=8, zorder=4)
    ax1.text(PRINCIPLED_EPS + 0.3, disconnected[pidx] + 3,
             f'ε = {PRINCIPLED_EPS}', color="tab:red", fontsize=14)
    ax1.set_xlabel('Epsilon ε', fontsize=15)
    ax1.set_ylabel('Disconnected RED nodes (%)', fontsize=15)
    ax1.set_ylim(0, 105)

    # Right: Mean CBTD per step
    ax2.plot(epsilon_values, mean_cbtd_steps, color="tab:green", linewidth=2,
             marker='o', markersize=5, markerfacecolor="tab:green", zorder=3)
    ax2.plot(PRINCIPLED_EPS, mean_cbtd_steps[pidx], 'o', color="tab:red",
             markersize=8, zorder=4)
    ax2.text(PRINCIPLED_EPS + 0.3, mean_cbtd_steps[pidx] - 0.6,
             f'ε = {PRINCIPLED_EPS}', color="tab:red", fontsize=14)
    ax2.set_xlabel('Epsilon ε', fontsize=15)
    ax2.set_ylabel('Mean distance per step (CBTD)', fontsize=15)
    ax2.set_ylim(0, None)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")


def plot_d_vs_d2(results, save_path):
    labels = list(results.keys())
    totals = [results[l]['single'] + results[l]['multi'] for l in labels]
    single = [results[l]['single'] / totals[i] * 100 for i, l in enumerate(labels)]
    multi  = [results[l]['multi']  / totals[i] * 100 for i, l in enumerate(labels)]

    x = np.arange(len(labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(5, 4))
    _style(ax)

    b1 = ax.bar(x - width / 2, single, width, label='Single-hop paths', color="tab:blue", zorder=3, linewidth=0, alpha=0.9)
    b2 = ax.bar(x + width / 2, multi, width, label='Multi-hop paths', color="tab:green", zorder=3, linewidth=0, alpha=0.9)

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f'{bar.get_height():.1f}%',
                ha='center', va='bottom', fontsize=13)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=15)
    ax.set_ylabel('Paths (%)', fontsize=15)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11, frameon=True, loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading nodes and building graph ...")
    nodes = create_nodes(DATASET_FOLDER)
    graph = Graph(nodes)
    n_red  = sum(1 for n in graph.nodes if n.color == Color.RED)
    n_blue = sum(1 for n in graph.nodes if n.color == Color.BLUE)
    print(f"  {len(graph.nodes)} nodes  |  {n_red} RED  |  {n_blue} BLUE")
    print(f"  principled ε = {graph.epsilon:.4f}\n")

    print("── Epsilon sensitivity sweep ──────────────────────────────")
    disconnected, mean_cbtd_steps = sweep_epsilon(graph, EPSILON_RANGE)
    plot_epsilon_sensitivity(
        EPSILON_RANGE, disconnected, mean_cbtd_steps,
        save_path="results/epsilon_sensitivity.png"
    )

    print("\n── d vs d² comparison ─────────────────────────────────────")
    d_results = compute_d_vs_d2(graph)
    plot_d_vs_d2(d_results, save_path="results/d_vs_d2.png")

    print("\nDone.")