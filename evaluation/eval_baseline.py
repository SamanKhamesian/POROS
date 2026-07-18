"""
eval_baseline.py — Endpoint-only CE baseline comparison.

Compares POROS against endpoint-only counterfactual explanation (CE) methods,
covering Wachter et al. 2017 and DiCE (Mothilal et al. 2020). These methods
produce a single target state — the nearest BLUE node by raw CBTD — and expect
the patient to close the full behavioral gap in a single step. POROS routes
through the ε-graph instead, using Dijkstra with d² edge weights.

For each source node, the nearest BLUE by raw CBTD is identified. This is the
target any endpoint-only method would nominate. The source is then classified
into one of four mutually exclusive cases:

  Case 1  CBTD ≤ ε  |  POROS 1-hop     →  both methods agree completely;
                                            same target, same single step
  Case 2  CBTD ≤ ε  |  POROS multi-hop  →  d² Dijkstra prefers two or more
                                            smaller hops over the single valid hop
  Case 3  CBTD > ε  |  POROS finds path →  endpoint-only jump infeasible;
                                            POROS routes through intermediate peers
  Case 4  CBTD > ε  |  POROS also fails  →  both methods cannot reach the target;
                                            jump statistics confirm behavioral isolation

Two populations are evaluated independently:
  red  :  RED source nodes only  (clinically primary — mirrors Query D)
  all  :  all source nodes       (complete picture  — mirrors Query B)

Functions
  compute_baseline  :  classify all source nodes; collect per-case statistics
  print_baseline    :  console report for both populations
"""

from collections import defaultdict

import numpy as np

from distance import DELTA
from node import Color, FEATURE_KEYS
from query import _dijkstra_all
from evaluation.eval_compute import reconstruct_path


# ── Style ─────────────────────────────────────────────────────────────────────

_DISPLAY_NAMES = {"avg_meal_bolus_delta_minutes": "avg_meal_bolus_delta_min"}
FEATURE_LABEL  = {k: _DISPLAY_NAMES.get(k, k) for k in FEATURE_KEYS}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stats(vals):
    if not vals:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(np.median(a)), float(a.min()), float(a.max()), float(a.std())


def _ascii_bar(value, max_value, width=12):
    filled = int(round(value / max_value * width)) if max_value > 0 else 0
    return "█" * filled + "░" * (width - filled)


# ── Computation ───────────────────────────────────────────────────────────────

def _make_pop():
    """Empty result container for one population (red or all)."""
    return {
        "n"    : 0,
        "case1": {"n": 0},
        "case2": {
            "n"     : 0,
            "direct": {"cbtd": [], "tir": [], "feat": defaultdict(list)},
            "poros" : {"hops": [], "step_tir": [], "step_cbtd": [],
                       "step_feat": defaultdict(list)},
        },
        "case3": {
            "n"     : 0,
            "direct": {"cbtd": [], "tir": [], "feat": defaultdict(list)},
            "poros" : {"hops": [], "step_tir": [], "step_cbtd": [],
                       "step_feat": defaultdict(list)},
        },
        "case4": {
            "n"     : 0,
            "direct": {"cbtd": [], "tir": [], "feat": defaultdict(list)},
        },
    }


def compute_baseline(graph):
    """
    Classify each source node against the endpoint-only baseline target
    (nearest BLUE by raw CBTD) and collect per-case statistics.

    Direct edge existence is determined from graph.edges — the exact same
    condition used during graph construction — rather than by recomputing
    CBTD vs ε, avoiding any floating-point inconsistency.

    CBTD values use graph.metric.pairwise (L2) throughout, matching the
    same metric used to build edges and compute ε.

    Parameters
    ----------
    graph : Graph
        Built ε-graph.  Required attributes:
          graph.epsilon         — ε threshold used during construction
          graph.nodes           — list of all Node objects
          graph.edges           — dict  node_id → [(target_id, weight_sq), ...]
          graph.node_index      — dict  node_id → Node
          graph.metric.pairwise — dict  (a_id, b_id) → raw L2 CBTD

    Returns
    -------
    dict with keys:
        "red"     : results for RED source nodes only
        "all"     : results for all source nodes
        "epsilon" : ε value from graph construction
    """
    epsilon    = graph.epsilon
    nodes      = graph.nodes
    blue_nodes = [nd for nd in nodes if nd.color == Color.BLUE]
    n          = len(nodes)

    red_pop = _make_pop()
    all_pop = _make_pop()

    for i, source in enumerate(nodes):
        print(f"\r  [{i+1:3d}/{n}]  {source.subject}  TIR={source.tir:.1f}%",
              end="", flush=True)

        valid = [bl for bl in blue_nodes if bl.tir > source.tir]
        if not valid:
            continue

        is_red = (source.color == Color.RED)

        # ── Endpoint-only target: nearest BLUE by L2 CBTD (matches graph) ────
        target_d = min(valid, key=lambda bl: graph.metric.pairwise[(source.node_id, bl.node_id)])
        cbtd_d   = graph.metric.pairwise[(source.node_id, target_d.node_id)]
        tir_d    = target_d.tir - source.tir
        feat_d   = {k: abs(target_d.features[k] - source.features[k]) / DELTA[k]
                    for k in FEATURE_KEYS}

        # ── POROS: Dijkstra from source ───────────────────────────────────────
        dist, prev = _dijkstra_all(graph, source.node_id)
        reachable  = target_d.node_id in dist

        # Direct edge check via graph structure — exact, no CBTD recomputation.
        # An edge source→target_d exists iff it was admitted during construction
        # (CBTD ≤ ε and target_d.tir > source.tir, both already satisfied here).
        direct_edge = any(b_id == target_d.node_id
                          for b_id, _ in graph.edges.get(source.node_id, []))

        # ── Classify into one of four cases ───────────────────────────────────
        if direct_edge and reachable:
            path = reconstruct_path(graph, prev, target_d.node_id)
            hops = len(path) - 1
            case = "case1" if hops == 1 else "case2"

        elif not direct_edge and reachable:
            # CBTD > ε: no direct edge, but a multi-hop graph path exists.
            path = reconstruct_path(graph, prev, target_d.node_id)
            hops = len(path) - 1
            case = "case3"

        else:
            # Case 4: no direct edge AND no graph path to target_d.
            # Note: direct_edge=True AND not reachable is impossible by
            # construction — a direct edge guarantees reachability.
            path = None
            hops = None
            case = "case4"

        # ── Accumulate into population containers ─────────────────────────────
        for pop, include in [(red_pop, is_red), (all_pop, True)]:
            if not include:
                continue

            pop["n"]       += 1
            pop[case]["n"] += 1

            if case in ("case2", "case3"):
                # endpoint-only side
                pop[case]["direct"]["cbtd"].append(cbtd_d)
                pop[case]["direct"]["tir"].append(tir_d)
                for k in FEATURE_KEYS:
                    pop[case]["direct"]["feat"][k].append(feat_d[k])

                # POROS side
                step_tir = [path[j+1].tir - path[j].tir for j in range(hops)]
                pop[case]["poros"]["hops"].append(hops)
                pop[case]["poros"]["step_tir"].extend(step_tir)
                for j in range(hops):
                    aa, bb = path[j], path[j + 1]
                    pop[case]["poros"]["step_cbtd"].append(
                        graph.metric.pairwise[(aa.node_id, bb.node_id)])
                    for k in FEATURE_KEYS:
                        pop[case]["poros"]["step_feat"][k].append(
                            abs(bb.features[k] - aa.features[k]) / DELTA[k])

            if case == "case4":
                # collect direct jump stats to confirm behavioral isolation
                pop[case]["direct"]["cbtd"].append(cbtd_d)
                pop[case]["direct"]["tir"].append(tir_d)
                for k in FEATURE_KEYS:
                    pop[case]["direct"]["feat"][k].append(feat_d[k])

    print()
    return {"red": red_pop, "all": all_pop, "epsilon": epsilon}


# ── Report ────────────────────────────────────────────────────────────────────

def _print_case_comparison(case_data, indent="  "):
    """
    Side-by-side table for case 2 or case 3.
    Left column  : endpoint-only (single direct jump).
    Right column : POROS (multi-hop graph path).
    Total TIR gain is identical for both — same source, same target.
    All CBTD values are L2, matching ε.
    """
    direct = case_data["direct"]
    poros  = case_data["poros"]

    dc_m,  dc_med, _, _, _       = _stats(direct["cbtd"])
    dt_m,  dt_med, _, _, dt_std  = _stats(direct["tir"])
    hm,    hmed,   _, _, _       = _stats(poros["hops"])
    st_m,  st_med, _, _, st_std  = _stats(poros["step_tir"])
    sc_m,  sc_med, _, _, _       = _stats(poros["step_cbtd"])

    d_feat = {k: float(np.mean(direct["feat"][k]))     if direct["feat"][k]     else 0.0
              for k in FEATURE_KEYS}
    p_feat = {k: float(np.mean(poros["step_feat"][k])) if poros["step_feat"][k] else 0.0
              for k in FEATURE_KEYS}

    ind  = f"{indent}  "
    L, N = 26, 16

    # helpers — each returns exactly N characters so columns stay aligned
    def _n(v):  return f"{v:>{N}.2f}"
    def _pp(v): return f"{v:>{N-3}.1f} pp"
    def _s(v):  return f"{v:>{N}}"

    print(f"{ind}{'':>{L}}  {'Endpoint-only':>{N}}  {'POROS':>{N}}")
    print(f"{ind}{'':>{L}}  {'(direct jump)':>{N}}  {'(graph path)':>{N}}")
    print(f"{ind}{'':─<{L}}  {'':─<{N}}  {'':─<{N}}")
    print(f"{ind}{'n cases':<{L}}  {_s(case_data['n'])}  {_s(case_data['n'])}")
    print(f"{ind}{'hops':<{L}}  {_s('1')}  {hm:>{N}.2f}  (median {hmed:.1f})")
    print(f"{ind}{'CBTD / step  mean':<{L}}  {_n(dc_m)}  {_n(sc_m)}")
    print(f"{ind}{'CBTD / step  median':<{L}}  {_n(dc_med)}  {_n(sc_med)}")
    print(f"{ind}{'TIR gain / step  mean':<{L}}  {_pp(dt_m)}  {_pp(st_m)}")
    print(f"{ind}{'TIR gain / step  std':<{L}}  {_pp(dt_std)}  {_pp(st_std)}")
    print(f"{ind}{'TIR gain / step  median':<{L}}  {_pp(dt_med)}  {_pp(st_med)}")
    print(f"{ind}{'total TIR gain':<{L}}  {_pp(dt_m)}  {_pp(dt_m)}")
    print()
    print(f"{ind}Behavioral change per step  (MCID units)")

    LABEL_W, BAR_W = 30, 10
    max_val = max(max(d_feat.values()), max(p_feat.values()), 1e-9)
    print(f"{ind}  {'Feature':<{LABEL_W}}  {'Endpoint':>8}  {'':>{BAR_W}}  {'POROS/step':>10}  {'':>{BAR_W}}")
    print(f"{ind}  {'':─<{LABEL_W}}  {'':─>8}  {'':─>{BAR_W}}  {'':─>10}  {'':─>{BAR_W}}")
    for k in FEATURE_KEYS:
        dv, pv = d_feat[k], p_feat[k]
        print(f"{ind}  {FEATURE_LABEL[k]:<{LABEL_W}}  "
              f"{dv:8.2f}  {_ascii_bar(dv, max_val, BAR_W)}  "
              f"{pv:10.2f}  {_ascii_bar(pv, max_val, BAR_W)}")


def _print_case4(case4_data, epsilon, indent="  "):
    """
    Single-column table for case 4 — both methods fail.
    Shows the endpoint-only jump stats to confirm behavioral isolation:
    the required CBTD is above ε and no intermediate path exists.
    """
    direct = case4_data["direct"]
    dc_m,  dc_med, _, _, _      = _stats(direct["cbtd"])
    dt_m,  dt_med, _, _, dt_std = _stats(direct["tir"])
    d_feat = {k: float(np.mean(direct["feat"][k])) if direct["feat"][k] else 0.0
              for k in FEATURE_KEYS}

    ind  = f"{indent}  "
    L, N = 26, 16

    def _n(v):  return f"{v:>{N}.2f}"
    def _pp(v): return f"{v:>{N-3}.1f} pp"
    def _s(v):  return f"{v:>{N}}"

    print(f"{ind}{'':>{L}}  {'Endpoint-only':>{N}}")
    print(f"{ind}{'':>{L}}  {'(direct jump)':>{N}}")
    print(f"{ind}{'':─<{L}}  {'':─<{N}}")
    print(f"{ind}{'n cases':<{L}}  {_s(case4_data['n'])}")
    print(f"{ind}{'CBTD  mean':<{L}}  {_n(dc_m)}")
    print(f"{ind}{'CBTD  median':<{L}}  {_n(dc_med)}")
    print(f"{ind}{'ε':<{L}}  {_n(epsilon)}")
    print(f"{ind}{'TIR gain  mean':<{L}}  {_pp(dt_m)}")
    print(f"{ind}{'TIR gain  std':<{L}}  {_pp(dt_std)}")
    print(f"{ind}{'TIR gain  median':<{L}}  {_pp(dt_med)}")
    print()
    print(f"{ind}Behavioral change required  (MCID units)")

    LABEL_W, BAR_W = 30, 12
    max_val = max(d_feat.values(), default=1e-9)
    print(f"{ind}  {'Feature':<{LABEL_W}}  {'Δv':>6}  {'':>{BAR_W}}")
    print(f"{ind}  {'':─<{LABEL_W}}  {'':─>6}  {'':─>{BAR_W}}")
    for k in FEATURE_KEYS:
        v = d_feat[k]
        print(f"{ind}  {FEATURE_LABEL[k]:<{LABEL_W}}  {v:6.2f}  {_ascii_bar(v, max_val, BAR_W)}")


def _print_population(pop, epsilon, indent="  "):
    """Full breakdown and case comparisons for one population."""
    n  = pop["n"]
    c1 = pop["case1"]["n"]
    c2 = pop["case2"]["n"]
    c3 = pop["case3"]["n"]
    c4 = pop["case4"]["n"]

    p  = indent
    sp = indent + "  "

    def pct(k): return f"{k / n * 100:.1f}%" if n > 0 else "—"

    # ── Coverage summary ──────────────────────────────────────────────────────
    covered = c1 + c2 + c3
    print(f"{p}POROS path coverage    {covered:>4} / {n}  ({pct(covered)})  — POROS finds a path")
    print(f"{p}Both methods fail      {c4:>4} / {n}  ({pct(c4)})  — behaviorally isolated nodes")
    print()

    # ── Case breakdown ────────────────────────────────────────────────────────
    print(f"{p}Case breakdown")
    print(f"{sp}Case 1  CBTD ≤ ε  +  POROS 1-hop      :  "
          f"{c1:>4} / {n}  ({pct(c1)})  both agree completely")
    print(f"{sp}Case 2  CBTD ≤ ε  +  POROS multi-hop  :  "
          f"{c2:>4} / {n}  ({pct(c2)})  d² prefers gradual steps")
    print(f"{sp}Case 3  CBTD > ε  +  POROS finds path :  "
          f"{c3:>4} / {n}  ({pct(c3)})  endpoint-only infeasible")
    print(f"{sp}Case 4  CBTD > ε  +  POROS also fails :  "
          f"{c4:>4} / {n}  ({pct(c4)})  both methods fail")

    # ── Case 2 ────────────────────────────────────────────────────────────────
    if c2 > 0:
        print(f"\n{p}── Case 2  CBTD ≤ ε  +  POROS multi-hop  (n={c2})  {'─' * 16}")
        print(f"{sp}A direct edge exists — the endpoint-only jump is valid (CBTD ≤ ε).")
        print(f"{sp}d² cost favours two or more smaller hops over the single valid hop.")
        _print_case_comparison(pop["case2"], indent)

    # ── Case 3 ────────────────────────────────────────────────────────────────
    if c3 > 0:
        print(f"\n{p}── Case 3  CBTD > ε  +  POROS finds path  (n={c3})  {'─' * 17}")
        print(f"{sp}No direct edge — endpoint-only jump exceeds ε = {epsilon:.2f}.")
        print(f"{sp}POROS routes through intermediate observed peers instead.")
        _print_case_comparison(pop["case3"], indent)

    # ── Cases 2 + 3 combined ──────────────────────────────────────────────────
    if c2 + c3 > 0:
        combined = {"n": c2 + c3, "direct": {"cbtd": (pop["case2"]["direct"]["cbtd"] + pop["case3"]["direct"]["cbtd"]),
            "tir": (pop["case2"]["direct"]["tir"] + pop["case3"]["direct"]["tir"]),
            "feat": {k: (pop["case2"]["direct"]["feat"][k] + pop["case3"]["direct"]["feat"][k]) for k in FEATURE_KEYS}, },
            "poros": {"hops": (pop["case2"]["poros"]["hops"] + pop["case3"]["poros"]["hops"]),
                "step_tir": (pop["case2"]["poros"]["step_tir"] + pop["case3"]["poros"]["step_tir"]),
                "step_cbtd": (pop["case2"]["poros"]["step_cbtd"] + pop["case3"]["poros"]["step_cbtd"]),
                "step_feat": {k: (pop["case2"]["poros"]["step_feat"][k] + pop["case3"]["poros"]["step_feat"][k]) for k in FEATURE_KEYS}, }, }

        print(f"\n{p}── Cases 2 + 3  POROS finds path  (n={c2 + c3})  {'─' * 20}")
        print(f"{sp}All cases where POROS decomposes into steps — regardless of feasibility.")
        print(f"{sp}Endpoint-only methods propose a single direct jump in all of these.")

        print(f"{sp}  Case 2  (CBTD ≤ ε, DiCE jump valid)     :  "
              f"{c2:>4} / {c2 + c3}  ({c2 / (c2 + c3) * 100:.1f}%)")

        print(f"{sp}  Case 3  (CBTD > ε, DiCE jump infeasible):  "
              f"{c3:>4} / {c2 + c3}  ({c3 / (c2 + c3) * 100:.1f}%)")
        _print_case_comparison(combined, indent)


    # ── Case 4 ────────────────────────────────────────────────────────────────
    if c4 > 0:
        print(f"\n{p}── Case 4  both methods fail  (n={c4})  {'─' * 28}")
        print(f"{sp}No direct edge and no graph path to the nominated target.")
        print(f"{sp}Jump statistics below confirm genuine behavioral isolation.")
        _print_case4(pop["case4"], epsilon, indent)


def print_baseline(bl):
    """
    Top-level console report for the endpoint-only baseline comparison.
    RED sources (clinically primary) are printed first, all sources second.
    """
    W       = 72
    SEP     = f"  {'─' * (W - 2)}"
    epsilon = bl["epsilon"]

    print(f"\n{'═' * W}")
    print("  Endpoint-only CE baseline comparison")
    print("  Wachter et al. 2017  |  DiCE (Mothilal et al. 2020)")
    print(f"{'═' * W}\n")

    # ── RED sources — primary ─────────────────────────────────────────────────
    red = bl["red"]
    print(SEP)
    print(f"  RED nodes → nearest BLUE   ({red['n']} sources)")
    print(SEP)
    _print_population(red, epsilon, indent="  ")

    # ── All sources — secondary ───────────────────────────────────────────────
    all_ = bl["all"]
    print(f"\n{SEP}")
    print(f"  All nodes → nearest BLUE   ({all_['n']} sources)")
    print(SEP)
    _print_population(all_, epsilon, indent="  ")

    print(f"\n{'═' * W}\n")