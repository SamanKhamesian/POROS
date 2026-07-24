"""
eval_face_pairs.py — FACE vs POROS over the all-pairs query set.

Companion to eval_face.py, not a replacement. eval_face.py lets FACE select its
own target; this file removes target selection from the comparison entirely.

Setting
───────────────────────────────────────────────────────────────────────────────
Targets come from the Query C pair set already used in the paper: every RED node
is a source, and every BLUE node with strictly higher TIR is a target. Both
methods are routed to the same target on the same pair, so the comparison
isolates path construction alone.

  POROS path : Dijkstra on the directed ε-graph, d² edge weights
  FACE path  : Dijkstra on the undirected k-NN graph, -log(p̂(mid))·d weights

A pair is kept only when both methods reach the target. All metrics are pooled
over kept pairs.

Routing to a specified target is within the FACE family's own scope — LocalFACE
(Small et al., 2023) constructs a path from a factual to a pre-identified
counterfactual endpoint.

Efficiency
───────────────────────────────────────────────────────────────────────────────
One Dijkstra per source per graph yields paths to all targets, so the cost is
P Dijkstras on each graph, where P is the number of RED sources.

Metrics
───────────────────────────────────────────────────────────────────────────────
  Feasibility (ε from the POROS derivation, applied to both methods)
    steps_over_eps       % of steps with d > ε                     POROS: 0
    paths_over_eps       % of paths with ≥ 1 step exceeding ε       POROS: 0
    max_step_d           largest single step in a path

  Monotonicity
    steps_backward       % of steps where TIR decreases             POROS: 0
    paths_backward       % of paths with ≥ 1 TIR-decreasing step    POROS: 0
    paths_monotone       % of paths monotone end to end             POROS: 100
    paths_below_source   % of paths dipping below the source's TIR  POROS: 0
    dip_below_source     magnitude of that dip in pp

  Step magnitude / structure
    hops, d_per_step, tir_per_step

  Grounding / density
    cross_patient, red_intermediate, path_density

Second block — POROS capability, no FACE column
───────────────────────────────────────────────────────────────────────────────
Queries with no class flip, undefined for classifier-driven counterfactual
methods. Reported as POROS-only counts and hop statistics.

  BLUE → BLUE   pairs with strictly higher TIR
  RED  → RED    pairs with strictly higher TIR

Run once per dataset by switching DATASET in config.py.

Functions
  _path_metrics            : per-path measurements, shared by both methods
  _accumulate              : fold one path into a population container
  compute_pairs_comparison : FACE vs POROS over the Query C pair set
  compute_poros_capability : BLUE→BLUE and RED→RED reachability, POROS only
  print_pairs_comparison   : console report, mean ± SD throughout
"""

import numpy as np

from config import DATASET, DATASET_FOLDER
from evaluation.eval_compute import reconstruct_path
from face_baseline import FaceGraph, face_shortest_paths
from graph import Graph, create_nodes
from node import Color
from query import _dijkstra_all


# ── FACE hyperparameters ──────────────────────────────────────────────────────
# k is FACE's own free parameter and carries no derivation. It is deliberately
# not tied to the POROS ε: the k-NN construction keeps the baseline graph free
# of any dependence on the ε derivation.

FACE_K       = 5
CONNECTIVITY = "knn"      # "knn" | "epsilon"
FACE_EPSILON = None       # only used if CONNECTIVITY == "epsilon"


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


def _face_reconstruct(fg, prev, target_id):
    """Rebuild a node-list path from a FACE Dijkstra prev-map."""
    path, cur = [], target_id
    while cur is not None:
        path.append(fg.node_index[cur])
        cur = prev.get(cur)
    path.reverse()
    return path


def _path_metrics(graph, path, epsilon, density=None):
    """
    Measure one path. Applied identically to a FACE path and a POROS path, using
    the POROS CBTD (graph.metric.pairwise) and the POROS ε in both cases, so the
    two methods are held to one common standard.

    Parameters
    ----------
    graph    : Graph        built POROS ε-graph (source of CBTD and ε)
    path     : list[Node]   ordered source → target
    epsilon  : float        ε threshold
    density  : dict | None  node_id → KDE density

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
    dip         = source.tir - min_tir            # > 0 means the path dropped below source

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


# ── Computation: FACE vs POROS over the Query C pair set ──────────────────────

def compute_pairs_comparison(graph, fg):
    """
    Route both methods to the same target on every RED → higher-TIR-BLUE pair.

    For each RED source, one Dijkstra is run on each graph, yielding paths to all
    reachable targets at once. A pair contributes to the metrics only when both
    methods reach the target.

    Returns
    -------
    dict with keys:
      "epsilon", "n_red", "n_pairs", "n_face_reach", "n_poros_reach",
      "n_kept", "face", "poros"
    """
    epsilon    = graph.epsilon
    red_nodes  = [n for n in graph.nodes if n.color == Color.RED]
    blue_nodes = [n for n in graph.nodes if n.color == Color.BLUE]
    n_red      = len(red_nodes)

    face  = _new_pop()
    poros = _new_pop()

    n_pairs       = 0
    n_face_reach  = 0
    n_poros_reach = 0
    n_kept        = 0

    for i, source in enumerate(red_nodes):
        print(f"\r  [{i+1:4d}/{n_red}]  {source.subject}  TIR={source.tir:.1f}%",
              end="", flush=True)

        targets = [b for b in blue_nodes if b.tir > source.tir]
        if not targets:
            continue

        f_dist, f_prev = face_shortest_paths(fg, source.node_id)
        p_dist, p_prev = _dijkstra_all(graph, source.node_id)

        for target in targets:
            n_pairs += 1
            tid = target.node_id

            in_face  = tid in f_dist
            in_poros = tid in p_dist

            if in_face:
                n_face_reach += 1
            if in_poros:
                n_poros_reach += 1
            if not (in_face and in_poros):
                continue

            n_kept += 1
            f_path = _face_reconstruct(fg, f_prev, tid)
            p_path = reconstruct_path(graph, p_prev, tid)

            _accumulate(face,  _path_metrics(graph, f_path, epsilon, fg.density))
            _accumulate(poros, _path_metrics(graph, p_path, epsilon, fg.density))

    print()
    return {
        "epsilon":       epsilon,
        "n_red":         n_red,
        "n_pairs":       n_pairs,
        "n_face_reach":  n_face_reach,
        "n_poros_reach": n_poros_reach,
        "n_kept":        n_kept,
        "face":          face,
        "poros":         poros,
    }


# ── Computation: POROS capability queries, no FACE column ─────────────────────

def compute_poros_capability(graph):
    """
    Queries with no class flip, undefined for classifier-driven CE methods.

      BLUE → BLUE   source and target both well-controlled, target higher TIR
      RED  → RED    source and target both poorly-controlled, target higher TIR

    Returns
    -------
    dict  label → {"n_pairs", "n_reached", "hops"}
    """
    blue = [n for n in graph.nodes if n.color == Color.BLUE]
    red  = [n for n in graph.nodes if n.color == Color.RED]

    out = {}

    for label, sources, targets in (("BLUE → BLUE", blue, blue),
                                    ("RED → RED",   red,  red)):
        n_pairs   = 0
        n_reached = 0
        hops      = []

        for i, source in enumerate(sources):
            print(f"\r  {label}  [{i+1:4d}/{len(sources)}]",
                  end="", flush=True)

            valid = [t for t in targets if t.tir > source.tir]
            if not valid:
                continue

            dist, prev = _dijkstra_all(graph, source.node_id)

            for target in valid:
                n_pairs += 1
                if target.node_id not in dist:
                    continue
                n_reached += 1
                path = reconstruct_path(graph, prev, target.node_id)
                hops.append(len(path) - 1)

        print()
        out[label] = {"n_pairs": n_pairs, "n_reached": n_reached, "hops": hops}

    return out


# ── Console report ────────────────────────────────────────────────────────────

def _row(label, face_val, poros_val, width=34):
    print(f"  {label:<{width}}  {face_val:>18}  {poros_val:>18}")


def _fmt_ms(vals, prec=2):
    mean, sd, *_ = _stats(vals)
    return f"{mean:.{prec}f} ± {sd:.{prec}f}"


def _summarize(pop):
    """Condense a population container into display-ready strings."""
    n_paths = pop["n_paths"]
    monotone = n_paths - pop["paths_backward"]
    return {
        "n_paths":        f"{n_paths}",
        "hops":           _fmt_ms(pop["hops"], 2),
        "d_per_step":     _fmt_ms(pop["d_steps"], 2),
        "tir_per_step":   _fmt_ms(pop["tir_steps"], 1),
        "max_step_d":     _fmt_ms(pop["max_step_d"], 2),
        "steps_over_eps": f"{_pct(pop['n_over_eps'], pop['n_steps']):.1f}%",
        "paths_over_eps": f"{_pct(pop['paths_over_eps'], n_paths):.1f}%",
        "steps_backward": f"{_pct(pop['n_backward'], pop['n_steps']):.1f}%",
        "paths_backward": f"{_pct(pop['paths_backward'], n_paths):.1f}%",
        "paths_monotone": f"{_pct(monotone, n_paths):.1f}%",
        "paths_below":    f"{_pct(pop['paths_below_source'], n_paths):.1f}%",
        "dip":            _fmt_ms(pop["dips"], 1),
        "cross":          f"{_pct(pop['cross'], n_paths):.1f}%",
        "red_inter":      f"{_pct(pop['n_red_inter'], pop['n_interior']):.1f}%",
        "density":        f"{_stats(pop['densities'])[0]:.3e}",
    }


def print_pairs_comparison(res, cap=None):
    """Full console report. All continuous metrics as mean ± SD."""
    print()
    print("═" * 80)
    print(f"  FACE vs POROS over the all-pairs query set  —  {DATASET}")
    print("═" * 80)

    print(f"\n  ε (POROS derivation)        : {res['epsilon']:.4f}")
    print(f"  FACE graph                  : {CONNECTIVITY}, k = {FACE_K}")

    n_pairs = res["n_pairs"]
    print(f"\n  Pair set  (RED source → BLUE target with higher TIR)")
    print(f"  {'─' * 76}")
    print(f"  RED sources                            : {res['n_red']}")
    print(f"  Candidate pairs                        : {n_pairs}")
    print(f"  Pairs FACE reaches                     : {res['n_face_reach']}  "
          f"({_pct(res['n_face_reach'], n_pairs):.1f}%)")
    print(f"  Pairs POROS reaches                    : {res['n_poros_reach']}  "
          f"({_pct(res['n_poros_reach'], n_pairs):.1f}%)")
    print(f"  Pairs kept (both reach)                : {res['n_kept']}  "
          f"({_pct(res['n_kept'], n_pairs):.1f}%)")

    f = _summarize(res["face"])
    p = _summarize(res["poros"])

    print(f"\n  Kept pairs — same source, same target")
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
    _row("paths monotone end to end (%)", f["paths_monotone"], p["paths_monotone"])
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
    print(f"\n  Table rows  ({DATASET}, n = {res['n_kept']} pairs)")
    print(f"  {'─' * 76}")
    print(f"  Steps exceeding ε (%)             {f['steps_over_eps']:>12}   {p['steps_over_eps']:>12}")
    print(f"  Paths with ≥1 step > ε (%)        {f['paths_over_eps']:>12}   {p['paths_over_eps']:>12}")
    print(f"  Steps decreasing TIR (%)          {f['steps_backward']:>12}   {p['steps_backward']:>12}")
    print(f"  Paths with ≥1 backward step (%)   {f['paths_backward']:>12}   {p['paths_backward']:>12}")
    print(f"  Paths monotone end to end (%)     {f['paths_monotone']:>12}   {p['paths_monotone']:>12}")
    print(f"  Mean hops per path                {f['hops']:>12}   {p['hops']:>12}")
    print(f"  Mean d per step                   {f['d_per_step']:>12}   {p['d_per_step']:>12}")
    print(f"  Mean TIR gain per step (pp)       {f['tir_per_step']:>12}   {p['tir_per_step']:>12}")
    print(f"  Cross-patient paths (%)           {f['cross']:>12}   {p['cross']:>12}")

    # ── POROS capability queries ──────────────────────────────────────────────
    if cap:
        print(f"\n  POROS capability — queries with no class flip (POROS only)")
        print(f"  {'─' * 76}")
        print(f"  {'Query':<16}  {'pairs':>12}  {'reached':>12}  {'rate':>8}  {'hops':>16}")
        print(f"  {'─' * 16}  {'─' * 12}  {'─' * 12}  {'─' * 8}  {'─' * 16}")
        for label, d in cap.items():
            rate = _pct(d["n_reached"], d["n_pairs"])
            print(f"  {label:<16}  {d['n_pairs']:>12}  {d['n_reached']:>12}  "
                  f"{rate:>7.1f}%  {_fmt_ms(d['hops'], 2):>16}")

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

    print("\nFACE vs POROS over the all-pairs query set ...")
    res = compute_pairs_comparison(graph, fg)

    print("\nPOROS capability queries ...")
    cap = compute_poros_capability(graph)

    print_pairs_comparison(res, cap)


if __name__ == "__main__":
    run()