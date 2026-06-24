"""
eval_main.py — Entry point. Builds the graph, runs all queries, prints the
               summary report, and generates figures.

Files
  eval_main.py     entry point — graph construction, query execution, figure calls
  eval_compute.py  computation only — all_pairs, nearest_paths, spearman; no I/O
  eval_report.py   output only — console tables, feature profiles, matplotlib figures

───────────────────────────────────────────────────────────────────────────────
Query types
  A. All-pairs     : source = any node,  target = any BLUE node with higher TIR
  B. Nearest-BLUE  : source = any node,  target = nearest BLUE node with higher TIR
  C. RED → BLUE    : source = RED node,  target = any BLUE node with higher TIR
  D. RED → nearest : source = RED node,  target = nearest BLUE node with higher TIR

  A and C are all-pairs queries     (one row per source-target pair).
  B and D are the nearest queries   (one target per source node).

Path types
  Direct path  : raw CBTD from source to target — no ε, no graph.
                 Baseline. Represents endpoint-only methods.
  Multi-hop    : Dijkstra path through the ε-graph (d² edge weights).
                 Our method. Per-step CBTD ≤ ε by construction.

Comparison columns for A and C
  Direct path         one value per pair  (raw CBTD jump, no ε)
  Multi-hop/step      one value per edge  (each edge in every path)
  Multi-hop/path avg  one value per path  (mean step size of each path)

  Example:
    Path 1:  steps = [5, 10]        → 2 hops
    Path 2:  steps = [3, 3, 3]      → 3 hops
    Path 3:  steps = [8, 2]         → 2 hops

    Multi-hop/step pools every individual edge across all paths into one flat list:
    All values: [5, 10, 3, 3, 3, 8, 2]   → 7 values
    Mean = 34 / 7 = 4.86 pp

    Multi-hop/path avg first computes each path's own mean, then averages those:
    Path means: [7.5,  3.0,  5.0]   → 3 values
    Mean = 15.5 / 3 = 5.17 pp

4 combinations for B and D
  For each source node, two targets are independently identified:
    target_d  nearest BLUE by minimum raw CBTD          (direct path's choice)
    target_g  nearest BLUE by minimum Dijkstra cost     (graph's choice)

  These two targets may or may not be the same node. The agreement rate
  (how often they coincide) is reported and is itself a finding.

  (1) direct path → target_d   trivially minimum CBTD; direct jumps to its own nearest
  (2) direct path → target_g   CBTD to the graph's recommendation
  (3) graph path  → target_d   can the graph reach direct's choice? at what step cost?
                                NOTE: target_d may not be reachable via the ε-graph
  (4) graph path  → target_g   trivially minimum path cost; graph navigates to its own nearest

  Reading the 4 combos:
    (1) vs (2)  how much further (in CBTD) is the graph's target vs the nearest node?
    (3) vs (4)  is the graph's target cheaper to navigate to than direct's target?
    (1) vs (4)  each method on its own terms — the core comparison
    agreement   when (1)=(2) and (3)=(4): both methods agree on the target
"""

from evaluation.eval_compute import compute_all_pairs, compute_nearest_paths
from evaluation.eval_report import print_summary
from graph import Graph, create_nodes
from node import Color
from viz import plot_path_length_histogram

DATASET_FOLDER = "dataset/ExActHealth"


def run():
    print("Loading nodes and building graph ...")
    nodes = create_nodes(DATASET_FOLDER)
    graph = Graph(nodes)

    n_red  = sum(1 for n in graph.nodes if n.color == Color.RED)
    n_blue = sum(1 for n in graph.nodes if n.color == Color.BLUE)
    print(f"  {len(graph.nodes)} nodes  |  {n_red} RED  |  {n_blue} BLUE\n")

    print("All-pairs  (A and C) ...")
    ap = compute_all_pairs(graph)

    print("\nNearest-BLUE  (B and D) ...")
    nr = compute_nearest_paths(graph)

    print_summary(ap, nr)

    print("\nGenerating P1 figures ...")
    plot_path_length_histogram(
        ap["a_mh_hops"],
        ap["c_mh_hops"],
        save_path="./results/path_length_histogram.png"
    )


if __name__ == "__main__":
    run()