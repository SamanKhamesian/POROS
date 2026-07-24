"""
eval_face.py — Path-aware baseline comparison: FACE (Poyiadzi et al., 2020) vs POROS.

Setting
───────────────────────────────────────────────────────────────────────────────
Every RED node (TIR < 70%) is a source. FACE selects its OWN target: the BLUE
node reachable at minimum density-weighted path cost over its undirected k-NN
graph. POROS is then routed to THAT SAME target through the ε-graph with d²
weights. Both methods therefore solve an identical problem — same source, same
destination — and differ only in how the path between them is constructed.

This mirrors the DiCE/Wachter comparison already reported for the endpoint-only
baselines: hold the target fixed, compare what each method returns as the route.

All path metrics are computed on the paired set only — the sources where both
methods produced a path. Identical source and identical target on both sides,
so every reported row compares like with like.

Reading the coverage number
───────────────────────────────────────────────────────────────────────────────
FACE returns a target for nearly every RED source, because nothing in its
construction constrains step size. POROS reaches a subset of those targets. That
subset is not a shortfall in POROS — it is the share of FACE recommendations for
which a certified-feasible, monotone route actually exists in the cohort. For the
remainder, FACE has nominated a destination it cannot certify a way to reach.
Reported here as "FACE targets POROS certifies".

Metrics computed (all of them — the paper selects a subset later)
───────────────────────────────────────────────────────────────────────────────
  Coverage
    face_target_rate     % of RED sources for which FACE returns a target
    poros_certified      % of FACE targets POROS reaches through the ε-graph

  Feasibility  (ε from POROS's derivation; applied identically to both methods)
    steps_over_eps       % of individual steps with d > ε          POROS: 0 by construction
    paths_over_eps       % of paths with ≥ 1 step exceeding ε      POROS: 0 by construction
    max_step_d           The largest single step in a path (mean ± SD over paths)

  Monotonicity
    steps_backward       % of individual steps where TIR decreases  POROS: 0 by construction
    paths_backward       % of paths with ≥ 1 TIR-decreasing step    POROS: 0 by construction
    paths_below_source   % of paths whose minimum TIR falls below the source's own TIR
    dip_below_source     magnitude of that dip in pp (mean ± SD, over dipping paths)

  Step magnitude / structure
    hops                 Number of hops per path
    d_per_step           CBTD per step
    tir_per_step         TIR gain per step (pp, signed — negatives are backward steps)

  Peer grounding
    cross_patient        % of paths touching a node from another subject
    red_intermediate     % of intermediate nodes that are RED
                         NOTE: this is NOT 0 for POROS and is not meant to be.
                         A POROS edge requires TIR improvement, not BLUE membership,
                         so a path may legitimately pass through higher-TIR RED nodes.
                         The structural POROS guarantee is paths_below_source = 0.

  Density (FACE's home turf — the quantity it explicitly optimizes and POROS does not)
    path_density         mean KDE density at nodes along the path

Run once per dataset by switching DATASET in config.py.

Functions
  _path_metrics          : per-path measurements, shared by both methods
  _accumulate            : fold one path's measurements into a population container
  compute_face_comparison: full evaluation over all RED sources
  print_face_comparison  : console report, mean ± SD throughout
"""

import numpy as np

from config import DATASET, DATASET_FOLDER
from evaluation.eval_compute import reconstruct_path
from face_baseline import FaceGraph, find_face_counterfactual
from graph import Graph, create_nodes
from node import Color
from query import _dijkstra_all


# ── FACE hyperparameters ──────────────────────────────────────────────────────
# k is FACE's own free parameter and carries no derivation. It is deliberately
# NOT tied to POROS's ε: the k-NN construction keeps the baseline graph free of
# any dependence on the ε derivation.

FACE_K       = 10
CONNECTIVITY = "knn"      # "knn" | "epsilon"
FACE_EPSILON = None       # only used if CONNECTIVITY == "epsilon"; never graph.epsilon


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stats(vals):
    """Return (mean, sd, median, min, max). Zeros on an empty list."""
    if not vals:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    a = np.array(vals, dtype=float)
    return (float(a.mean()), float(a.std()), float(np.median(a)),
            float(a.min()), float(a.max()))


def _pct(num, den):
    return 100.0 * num / den if den else 0.0


def _path_metrics(graph, path, epsilon, density=None):
    """
    Measure one path. Applies identically to a FACE path and a POROS path, using
    POROS's CBTD (graph.metric.pairwise) and POROS's ε in both cases, so the two
    methods are held to one common standard.

    Parameters
    ----------
    graph    : Graph        built POROS ε-graph (source of CBTD and ε)
    path     : list[Node]   ordered source → target
    epsilon  : float        ε threshold
    density  : dict | None  node_id → KDE density, for the density metric

    Returns
    -------
    dict of per-path measurements
    """
    source = path[0]
    hops   = len(path) - 1

    d_steps   = []
    tir_steps = []

    for j in range(hops):
        a, b = path[j], path[j + 1]
        d_steps.append(graph.metric.pairwise[(a.node_id, b.node_id)])
        tir_steps.append(b.tir - a.tir)

    n_over_eps  = sum(1 for d in d_steps if d > epsilon)
    n_backward  = sum(1 for t in tir_steps if t < 0)
    min_tir     = min(nd.tir for nd in path)
    dip         = source.tir - min_tir            # > 0 means path went below source

    interior    = path[1:-1]
    n_red_inter = sum(1 for nd in interior if nd.color == Color.RED)

    dens = ([density[nd.node_id] for nd in path]
            if density is not None else [])

    return {
        "hops":         hops,
        "d_steps":      d_steps,
        "tir_steps":    tir_steps,
        "n_over_eps":   n_over_eps,
        "n_backward":   n_backward,
        "max_step_d":   max(d_steps),
        "dip":          dip,
        "cross":        any(nd.subject != source.subject for nd in path),
        "n_interior":   len(interior),
        "n_red_inter":  n_red_inter,
        "mean_density": float(np.mean(dens)) if dens else 0.0,
    }


def _new_pop():
    """Empty accumulator for one method."""
    return {
        "n_paths":      0,
        "hops":         [],
        "d_steps":      [],   # flat pool across all paths
        "tir_steps":    [],   # flat pool across all paths
        "max_step_d":   [],   # one per path
        "dips":         [],   # one per path, only when > 0
        "densities":    [],   # one per path
        "n_steps":      0,
        "n_over_eps":   0,
        "n_backward":   0,
        "paths_over_eps":     0,
        "paths_backward":     0,
        "paths_below_source": 0,
        "cross":        0,
        "n_interior":   0,
        "n_red_inter":  0,
    }


def _accumulate(pop, m):
    """Fold one path's measurements into a population container."""
    pop["n_paths"]    += 1
    pop["hops"].append(m["hops"])
    pop["d_steps"].extend(m["d_steps"])
    pop["tir_steps"].extend(m["tir_steps"])
    pop["max_step_d"].append(m["max_step_d"])
    pop["densities"].append(m["mean_density"])

    pop["n_steps"]    += m["hops"]
    pop["n_over_eps"] += m["n_over_eps"]
    pop["n_backward"] += m["n_backward"]

    if m["n_over_eps"] > 0:
        pop["paths_over_eps"] += 1
    if m["n_backward"] > 0:
        pop["paths_backward"] += 1
    if m["dip"] > 0:
        pop["paths_below_source"] += 1
        pop["dips"].append(m["dip"])
    if m["cross"]:
        pop["cross"] += 1

    pop["n_interior"]  += m["n_interior"]
    pop["n_red_inter"] += m["n_red_inter"]


# ── Computation ───────────────────────────────────────────────────────────────

def compute_face_comparison(graph, fg):
    """
    Compare FACE and POROS on the RED → FACE-selected-target setting.

    For every RED source:
      1. FACE selects its target and returns its density-weighted path.
      2. POROS is routed to that same target through the ε-graph (d² weights).
      3. Both paths are measured against POROS's CBTD and ε.

    A source contributes to the path metrics only when both methods produced a
    path, so every reported row compares like with like.

    Returns
    -------
    dict with keys:
      "epsilon", "n_red", "n_face_target", "n_poros_certified", "face", "poros"
    """
    epsilon   = graph.epsilon
    red_nodes = [n for n in graph.nodes if n.color == Color.RED]
    n_red     = len(red_nodes)

    face  = _new_pop()
    poros = _new_pop()

    n_face_target     = 0
    n_poros_certified = 0

    for i, source in enumerate(red_nodes):
        print(f"\r  [{i+1:4d}/{n_red}]  {source.subject}  TIR={source.tir:.1f}%",
              end="", flush=True)

        # ── FACE: selects its own target, returns its own path ────────────────
        cf = find_face_counterfactual(fg, source.node_id)
        if cf is None:
            continue

        n_face_target += 1
        target_id = cf["target"].node_id

        # ── POROS: routed to the SAME target through the ε-graph ──────────────
        dist, prev = _dijkstra_all(graph, source.node_id)
        if target_id not in dist:
            continue

        n_poros_certified += 1
        poros_path = reconstruct_path(graph, prev, target_id)

        _accumulate(face,  _path_metrics(graph, cf["path"], epsilon, fg.density))
        _accumulate(poros, _path_metrics(graph, poros_path, epsilon, fg.density))

    print()
    return {
        "epsilon":           epsilon,
        "n_red":             n_red,
        "n_face_target":     n_face_target,
        "n_poros_certified": n_poros_certified,
        "face":              face,
        "poros":             poros,
    }


# ── Console report ────────────────────────────────────────────────────────────

def _row(label, face_val, poros_val, width=34):
    print(f"  {label:<{width}}  {face_val:>18}  {poros_val:>18}")


def _fmt_ms(vals, prec=2):
    mean, sd, *_ = _stats(vals)
    return f"{mean:.{prec}f} ± {sd:.{prec}f}"


def _summarize(pop):
    """Condense a population container into display-ready strings."""
    return {
        "n_paths":        f"{pop['n_paths']}",
        "hops":           _fmt_ms(pop["hops"], 2),
        "d_per_step":     _fmt_ms(pop["d_steps"], 2),
        "tir_per_step":   _fmt_ms(pop["tir_steps"], 1),
        "max_step_d":     _fmt_ms(pop["max_step_d"], 2),
        "steps_over_eps": f"{_pct(pop['n_over_eps'], pop['n_steps']):.1f}%",
        "paths_over_eps": f"{_pct(pop['paths_over_eps'], pop['n_paths']):.1f}%",
        "steps_backward": f"{_pct(pop['n_backward'], pop['n_steps']):.1f}%",
        "paths_backward": f"{_pct(pop['paths_backward'], pop['n_paths']):.1f}%",
        "paths_below":    f"{_pct(pop['paths_below_source'], pop['n_paths']):.1f}%",
        "dip":            _fmt_ms(pop["dips"], 1),
        "cross":          f"{_pct(pop['cross'], pop['n_paths']):.1f}%",
        "red_inter":      f"{_pct(pop['n_red_inter'], pop['n_interior']):.1f}%",
        "density":        f"{_stats(pop['densities'])[0]:.3e}",
    }


def print_face_comparison(res):
    """Full console report. All continuous metrics as mean ± SD."""
    print()
    print("═" * 80)
    print(f"  FACE vs POROS  —  {DATASET}")
    print("═" * 80)

    n_red   = res["n_red"]
    n_face  = res["n_face_target"]
    n_poros = res["n_poros_certified"]

    print(f"\n  ε (POROS derivation)        : {res['epsilon']:.4f}")
    print(f"  FACE graph                  : {CONNECTIVITY}, k = {FACE_K}")

    print(f"\n  Coverage")
    print(f"  {'─' * 76}")
    print(f"  RED sources                            : {n_red}")
    print(f"  FACE returns a target                  : {n_face}  "
          f"({_pct(n_face, n_red):.1f}%)")
    print(f"  FACE targets POROS certifies           : {n_poros}  "
          f"({_pct(n_poros, n_face):.1f}% of FACE targets, "
          f"{_pct(n_poros, n_red):.1f}% of RED sources)")
    print(f"  FACE targets with no feasible route    : {n_face - n_poros}  "
          f"({_pct(n_face - n_poros, n_face):.1f}% of FACE targets)")

    f = _summarize(res["face"])
    p = _summarize(res["poros"])

    print(f"\n  Paired paths — same source, same target")
    print(f"  {'─' * 76}")
    _row("", "FACE", "POROS")
    print(f"  {'─' * 76}")

    _row("paths", f["n_paths"], p["n_paths"])

    print(f"  {'· feasibility (ε)':<34}")
    _row("steps exceeding ε (%)",        f["steps_over_eps"], p["steps_over_eps"])
    _row("paths with ≥1 step > ε (%)",   f["paths_over_eps"], p["paths_over_eps"])
    _row("max step d in path",           f["max_step_d"],     p["max_step_d"])

    print(f"  {'· monotonicity':<34}")
    _row("steps decreasing TIR (%)",     f["steps_backward"], p["steps_backward"])
    _row("paths with ≥1 backward (%)",   f["paths_backward"], p["paths_backward"])
    _row("paths below source TIR (%)",   f["paths_below"],    p["paths_below"])
    _row("dip below source (pp)",        f["dip"],            p["dip"])

    print(f"  {'· step magnitude / structure':<34}")
    _row("hops per path",                f["hops"],           p["hops"])
    _row("d per step",                   f["d_per_step"],     p["d_per_step"])
    _row("TIR gain per step (pp)",       f["tir_per_step"],   p["tir_per_step"])

    print(f"  {'· grounding / density':<34}")
    _row("cross-patient paths (%)",      f["cross"],          p["cross"])
    _row("RED intermediate nodes (%)",   f["red_inter"],      p["red_inter"])
    _row("mean KDE density on path",     f["density"],        p["density"])

    # ── Paper-ready rows ──────────────────────────────────────────────────────
    print(f"\n  Table rows  ({DATASET}, n = {res['face']['n_paths']})")
    print(f"  {'─' * 76}")
    print(f"  Feasible coverage (%)             {_pct(n_poros, n_face):>10.1f}"
          f"   {100.0:>10.1f}")
    print(f"  Mean hops per path                {f['hops']:>10}   {p['hops']:>10}")
    print(f"  Mean d per step                   {f['d_per_step']:>10}   {p['d_per_step']:>10}")
    print(f"  Max d in path                     {f['max_step_d']:>10}   {p['max_step_d']:>10}")
    print(f"  Steps exceeding ε (%)             {f['steps_over_eps']:>10}   {p['steps_over_eps']:>10}")
    print(f"  Paths with ≥1 step > ε (%)        {f['paths_over_eps']:>10}   {p['paths_over_eps']:>10}")
    print(f"  Steps decreasing TIR (%)          {f['steps_backward']:>10}   {p['steps_backward']:>10}")
    print(f"  Paths with ≥1 backward step (%)   {f['paths_backward']:>10}   {p['paths_backward']:>10}")
    print(f"  Mean TIR gain per step (pp)       {f['tir_per_step']:>10}   {p['tir_per_step']:>10}")
    print(f"  Cross-patient paths (%)           {f['cross']:>10}   {p['cross']:>10}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def run():
    print(f"Loading nodes and building POROS graph  ({DATASET}) ...")
    nodes = create_nodes(DATASET_FOLDER)
    graph = Graph(nodes)

    n_red  = sum(1 for n in graph.nodes if n.color == Color.RED)
    n_blue = sum(1 for n in graph.nodes if n.color == Color.BLUE)
    print(f"  {len(graph.nodes)} nodes  |  {n_red} RED  |  {n_blue} BLUE")
    print(f"  ε = {graph.epsilon:.4f}\n")

    print("Building FACE graph ...")
    fg = FaceGraph(
        nodes,
        metric=graph.metric,          # identical CBTD on both sides
        connectivity=CONNECTIVITY,
        k=FACE_K,
        face_epsilon=FACE_EPSILON,
    )
    fg.summary()

    print("\nEvaluating FACE vs POROS ...")
    res = compute_face_comparison(graph, fg)
    print_face_comparison(res)


if __name__ == "__main__":
    run()