"""
──────────────────────────────────────────────────────────────────────────────
Exploratory visualizations for the counterfactual graph.
Each function saves its own PNG.  Run standalone or call individually.
──────────────────────────────────────────────────────────────────────────────
"""

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from graph import Graph, create_nodes

dataset_folder = "dataset/ExActHealth"

RED_C  = "tab:red"
BLUE_C = "tab:blue"
TEAL_C = "tab:green"
EPS_C  = "black"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "grid.linestyle":   "--",
    "grid.linewidth":   0.5,
    "font.size":        11,
})

def _box():
    """Annotation box styled like a matplotlib legend frame."""
    return dict(facecolor="white", edgecolor="gray", alpha=0.8)


# ── 1. TIR distribution ──────────────────────────────────────────

def plot_tir_distribution(graph: Graph) -> None:
    tirs = np.array([n.tir for n in graph.nodes])
    total = len(tirs)

    bin_edges = np.arange(0, 101, 10)
    counts, _ = np.histogram(tirs, bins=bin_edges)
    pcts = counts / total * 100

    labels = [f"{e}-{e + 10}" for e in bin_edges[:-1]]
    x_pos = np.arange(len(pcts))

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.bar(x_pos, pcts, width=0.75, color="tab:blue", alpha=0.6, edgecolor="none")
    ax.bar(x_pos, pcts, width=0.75, color="none", edgecolor="tab:blue", linewidth=2)

    for x, pct in zip(x_pos, pcts):
        if pct >= 0.5:
            ax.text(x, pct + 1.5, f"{pct:.1f}%", ha="center", va="bottom", fontsize=12, color="dimgray")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=14)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_xlabel("Time in Range (%)", fontsize=15)
    ax.set_ylabel("% of Nodes (Days)", fontsize=15)
    # ax.set_title("Node TIR Distribution", fontsize=16)
    ax.set_ylim(0, 50)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.2, color="gray")
    ax.set_facecolor("whitesmoke")

    plt.tight_layout()
    plt.savefig("results/graph_node_tir_distribution.png", dpi=300)
    plt.show()

# ── 2. edges distance distribution ──────────────────────────────────────────

def plot_edges_distance_distribution(graph: Graph) -> None:
    edge_dists = [
        graph.metric.pairwise[(src, tgt)]
        for src, neighbors in graph.edges.items()
        for tgt, _ in neighbors
    ]

    dists = np.array(edge_dists)
    total = len(dists)

    bin_edges = np.arange(0, np.ceil(dists.max()) + 1, 1)
    counts, _ = np.histogram(dists, bins=bin_edges)
    pcts      = counts / total * 100

    labels = [f"{int(e)}-{int(e)+1}" for e in bin_edges[:-1]]
    x_pos  = np.arange(len(pcts))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(x_pos, pcts, width=0.75, color="tab:green", alpha=0.4, edgecolor="none")
    ax.bar(x_pos, pcts, width=0.75, color="none", edgecolor="tab:green", linewidth=2)

    for x, pct in zip(x_pos, pcts):
        if pct >= 0.5:
            ax.text(x, pct + 1.5, f"{pct:.1f}%",
                    ha="center", va="bottom", fontsize=12, color="dimgray")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
    ax.set_xlabel("Distance (CBTD)", fontsize=13)
    ax.set_ylabel("% of Edges", fontsize=13)
    ax.set_title("Graph Edges Distance Distribution", fontsize=16)
    ax.set_ylim(0, 30)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.2, color="gray")
    ax.set_facecolor("whitesmoke")

    plt.tight_layout()
    plt.savefig("results/graph_edges_distance_distribution.png", dpi=300)
    plt.show()

# ── 3. all pairs distance distribution ──────────────────────────────────────────

def plot_pairs_distance_distribution(graph: Graph) -> None:
    all_dists = list(graph.metric.pairwise.values())
    dists = np.array(all_dists)
    epsilon = 10.95

    kde = gaussian_kde(dists, bw_method="scott")
    x_grid = np.linspace(0, dists.max(), 1000)
    y_kde = kde(x_grid)

    dx = x_grid[1] - x_grid[0]
    y_pct = y_kde * dx * 100

    left = x_grid <= epsilon
    right = x_grid >= epsilon

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(x_grid[left], y_pct[left], alpha=0.35, color="tab:blue")
    ax.fill_between(x_grid[right], y_pct[right], alpha=0.35, color="tab:red")
    ax.plot(x_grid[left], y_pct[left], color="tab:blue", linewidth=2)
    ax.plot(x_grid[right], y_pct[right], color="tab:red", linewidth=2)

    ax.axvline(x=epsilon, color="black", linestyle="--", linewidth=1.5)

    y_label = max(y_pct) * 1.12
    ax.text(epsilon + 0.8, y_label, f"ε = {epsilon}", fontsize=13, color="black", va="bottom")

    stats_text = (f"Mean:    {dists.mean():.2f}\n"
                  f"Median:  {np.median(dists):.2f}\n"
                  f"Min:      {dists.min():.2f}\n"
                  f"Max:    {dists.max():.2f}")

    ax.text(0.97,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=14,
            va="top",
            ha="right",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.8))

    tick_vals = np.arange(0, dists.max(), 10)
    ax.set_xticks(tick_vals)
    ax.set_xticklabels([str(int(v)) for v in tick_vals], fontsize=14)
    ax.tick_params(axis="y", labelsize=13)

    ax.set_xlabel("Distance (CBTD)", fontsize=16)
    ax.set_ylabel("% of Node Pairs", fontsize=16)
    ax.set_title("All-Pairs Distance Distribution", fontsize=18)
    ax.set_ylim(0, max(y_pct) * 1.25)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.2, color="gray")
    ax.set_facecolor("whitesmoke")

    plt.tight_layout()
    plt.savefig("results/graph_all_pairs_distance_distribution.png", dpi=300)
    plt.show()

# ── 4. path length histogram ──────────────────────────────────────────

def plot_path_length_histogram(a_mh_hops, c_mh_hops, save_path):
    """
    Distribution of multi-hop path lengths (in hops).
    Saves two separate figures:
      <base>_all.<ext>
      <base>_poorly_controlled.<ext>

    Parameters
    ----------
    a_mh_hops   : list[int]  — hop counts, all-pairs multi-hop
    c_mh_hops   : list[int]  — hop counts, poorly-controlled source multi-hop
    save_path   : str        — base path; suffixes are appended before the extension
    """
    import os

    base, ext = os.path.splitext(save_path)

    datasets = [(a_mh_hops, "tab:red", "All Source Nodes — Path Length Distribution", f"{base}_all{ext}"),
                (c_mh_hops, "tab:red", "Poorly-Controlled Sources — Path Length Distribution", f"{base}_poorly_controlled{ext}"), ]

    for hops, color, subtitle, path in datasets:
        fig, ax = plt.subplots(figsize=(8, 4.5))

        counts = Counter(hops)
        max_hop = max(counts)
        xs_int = list(range(2, max_hop + 1))
        total = sum(counts.values())
        pcts = [counts.get(x, 0) / total * 100 for x in xs_int]
        x_pos = np.arange(len(xs_int))
        labels = [str(x) for x in xs_int]

        ax.bar(x_pos, pcts, width=0.75, color=color, alpha=0.6, edgecolor="none")
        ax.bar(x_pos, pcts, width=0.75, color="none", edgecolor=color, linewidth=2)

        for x, pct in zip(x_pos, pcts):
            if pct >= 0.5:
                ax.text(x, pct + 0.8, f"{pct:.1f}%", ha="center", va="bottom", fontsize=12, color="dimgray")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=14)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlabel("Path length (Hops)", fontsize=15)
        ax.set_ylabel("% of Paths", fontsize=15)
        # ax.set_title(subtitle, fontsize=16)
        ax.set_ylim(0, max(pcts) * 1.25)
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.2, color="gray")
        ax.set_facecolor("whitesmoke")

        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  saved → {path}")
        plt.close(fig)


if __name__ == "__main__":
    nodes = create_nodes(dataset_folder)
    graph = Graph(nodes)

    plot_tir_distribution(graph)
    plot_edges_distance_distribution(graph)
    plot_pairs_distance_distribution(graph)