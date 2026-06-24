"""
──────────────────────────────────────────────────────────────────────────────
Exploratory visualizations for the counterfactual graph.
Each function saves its own PNG.  Run standalone or call individually.
──────────────────────────────────────────────────────────────────────────────
"""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
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


# ── 1. TIR distribution — KDE curve ──────────────────────────────────────────

def plot_tir_distribution(graph: Graph) -> None:
    tirs = np.array([n.tir for n in graph.nodes])
    total = len(tirs)

    bin_edges = np.arange(0, 101, 10)
    counts, _ = np.histogram(tirs, bins=bin_edges)
    pcts = counts / total * 100

    labels = [f"{e}-{e + 10}" for e in bin_edges[:-1]]
    x_pos = np.arange(len(pcts))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(x_pos, pcts, width=0.75, color="tab:blue", alpha=0.4, edgecolor="none")
    ax.bar(x_pos, pcts, width=0.75, color="none", edgecolor="tab:blue", linewidth=2)

    for x, pct in zip(x_pos, pcts):
        if pct >= 0.5:
            ax.text(x, pct + 1.5, f"{pct:.1f}%", ha="center", va="bottom", fontsize=12, color="dimgray")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
    ax.set_xlabel("Time in Range (%)", fontsize=13)
    ax.set_ylabel("% of Nodes (Days)", fontsize=13)
    ax.set_title("Node TIR Distribution", fontsize=16)
    ax.set_ylim(0, 50)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.2, color="gray")
    ax.set_facecolor("whitesmoke")

    plt.tight_layout()
    plt.savefig("basic_results/node_tir_distribution.png", dpi=200)
    plt.show()

# ── 2. TIR per patient — ridgeline ───────────────────────────────────────────

def plot_tir_per_patient(graph: Graph) -> None:
    """
    Ridgeline plot: one KDE ridge per patient, sorted descending by mean TIR.
    Within each ridge the fill is split at 70 %: RED below, BLUE above.
    """
    subject_nodes: dict = defaultdict(list)
    for node in graph.nodes:
        subject_nodes[node.subject].append(node)

    subjects = sorted(
        subject_nodes.keys(),
        key=lambda s: np.mean([n.tir for n in subject_nodes[s]]),
        reverse=True,
    )
    n = len(subjects)
    x = np.linspace(15, 105, 400)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.grid(False)

    for i, subject in enumerate(subjects):
        tirs = np.array([nd.tir for nd in subject_nodes[subject]])
        if len(tirs) < 2:
            continue

        kfn   = gaussian_kde(tirs, bw_method=0.45)
        y_raw = kfn(x)
        y_nrm = y_raw / y_raw.max() * 0.82
        base  = float(n - 1 - i)

        mask_red  = x <= 70
        mask_blue = x >= 70

        ax.fill_between(x[mask_red],  base, base + y_nrm[mask_red],
                        color=RED_C,  alpha=0.40, lw=0)
        ax.fill_between(x[mask_blue], base, base + y_nrm[mask_blue],
                        color=BLUE_C, alpha=0.40, lw=0)
        ax.plot(x, base + y_nrm, color="#555", lw=1.2)
        ax.hlines(base, 15, 105, colors="#dddddd", lw=0.5, zorder=0)

    ax.axvline(70, color="#555", lw=1.2, ls="--", alpha=0.8)
    ax.text(71, n - 0.1, "70 %", fontsize=8.5, color="#555", va="top")

    ax.set_yticks(range(n))
    ax.set_yticklabels(list(reversed(subjects)), fontsize=8.5)
    ax.set_xlabel("TIR (%)")
    ax.set_title("TIR Distribution per Patient  (sorted by mean TIR)")
    ax.set_xlim(15, 105)
    ax.set_ylim(-0.2, n)

    plt.tight_layout()
    plt.savefig("basic_results/plot_2_tir_per_patient.png", dpi=150)
    plt.show()
    print("Saved: plot_2_tir_per_patient.png")


# ── 3. Consecutive-day distances per patient — seaborn boxplot ───────────────

def plot_consec_distances_per_patient(graph: Graph) -> None:
    """
    Seaborn-style boxplot of within-patient consecutive-day distances.
    Mean (red dot), median (blue line), and std (vertical error bar) per patient.
    Dashed horizontal line at epsilon.
    """
    subject_nodes: dict = defaultdict(list)
    for node in graph.nodes:
        subject_nodes[node.subject].append(node)

    labels, data = [], []
    for subject in sorted(subject_nodes.keys()):
        s_nodes = sorted(subject_nodes[subject], key=lambda n: n.date)
        dists = [
            graph.metric.pairwise[(s_nodes[i].node_id, s_nodes[i + 1].node_id)]
            for i in range(len(s_nodes) - 1)
        ]
        if dists:
            labels.append(subject)
            data.append(np.array(dists))

    rows = [
        {"subject": subject, "distance": float(d)}
        for subject, dists in zip(labels, data)
        for d in dists
    ]
    df_plot = pd.DataFrame(rows)

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))

    sns.boxplot(
        data=df_plot,
        x="subject",
        y="distance",
        color="skyblue",
        showmeans=True,
        showfliers=False,
        meanprops={"marker": "o", "markerfacecolor": "red",
                   "markeredgecolor": "black", "markersize": 6},
        medianprops={"color": "blue", "linewidth": 1.5},
        ax=ax,
    )

    # Std error bars centred on the mean
    for pi, dists in enumerate(data):
        ax.errorbar(pi, dists.mean(), yerr=dists.std(),
                    fmt="none", color="black", capsize=3,
                    linewidth=1.0, zorder=5)

    ax.axhline(graph.epsilon, color="black", ls="--", lw=1.5)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Mean",
               markerfacecolor="red", markeredgecolor="black", markersize=8),
        Line2D([0], [0], color="blue",  lw=2,   label="Median"),
        Line2D([0], [0], color="black", lw=1.5, ls="--",
               label=f"ε = {graph.epsilon:.3f}"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    ax.set_title("Consecutive-Day Distances per Patient", fontsize=14)
    ax.set_xlabel("Patient", fontsize=12)
    ax.set_ylabel("Consecutive-day distance (CBTD)", fontsize=12)
    ax.tick_params(axis="x", labelsize=9, rotation=45)
    ax.tick_params(axis="y", labelsize=10)

    plt.tight_layout()
    plt.savefig("basic_results/plot_3_consec_per_patient.png", dpi=150)
    plt.show()
    print("Saved: plot_3_consec_per_patient.png")


def plot_distance_distribution(graph: Graph) -> None:
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
    ax.set_ylabel("% of Connections", fontsize=13)
    ax.set_title("Paired Nodes Distance Distribution", fontsize=16)
    ax.set_ylim(0, 30)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.2, color="gray")
    ax.set_facecolor("whitesmoke")

    plt.tight_layout()
    plt.savefig("basic_results/node_distance_distribution.png", dpi=200)
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    nodes = create_nodes(dataset_folder)
    graph = Graph(nodes)

    plot_tir_distribution(graph)
    plot_tir_per_patient(graph)
    plot_consec_distances_per_patient(graph)
    plot_distance_distribution(graph)