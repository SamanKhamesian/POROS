"""
eval_compute.py — All computation functions. No printing, no plotting.

Functions
  reconstruct_path      : rebuilds a node-list path from Dijkstra prev-map
  _raw_cbtd             : raw CBTD between two nodes, no ε constraint
  compute_all_pairs     : Queries A and C — direct path pass + multi-hop pass
  compute_nearest_paths : Queries B and D — 4-combo nearest-BLUE analysis
  compute_spearman      : Spearman ρ between edge distance and TIR gap
"""

import math
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

from distance import DELTA
from graph import Graph
from node import Color, FEATURE_KEYS
from query import _dijkstra_all


# ── Helpers ───────────────────────────────────────────────────────────────────

def reconstruct_path(graph, prev, target_id):
    path, cur = [], target_id
    while cur is not None:
        path.append(graph.node_index[cur])
        cur = prev[cur]
    path.reverse()
    return path


def _raw_cbtd(node_a, node_b):
    return sum(
        abs(node_b.features[k] - node_a.features[k]) / DELTA[k]
        for k in FEATURE_KEYS
    )


# ── Computation ───────────────────────────────────────────────────────────────

def compute_all_pairs(graph):
    """
    A and C: all-pairs queries.
    Target is always a BLUE node with strictly higher TIR than source.

    Direct path pass: iterates all source-target pairs, no ε, no graph.
    Multi-hop pass:   Dijkstra through ε-graph, records paths of 2+ edges.

    A: source = any node.   C: source = RED node (subset of A).
    """
    # A
    a_sh_tir        = [];  a_mh_flat = [];  a_mh_path_means = []
    a_mh_hops       = [];  a_sh_feat = defaultdict(list)
    a_mh_feat       = defaultdict(list)
    a_cross_sh      = 0;   a_cross_mh = 0

    # C
    c_sh_tir        = [];  c_mh_flat = [];  c_mh_path_means = []
    c_mh_hops       = [];  c_sh_feat = defaultdict(list)
    c_mh_feat       = defaultdict(list)
    c_cross_sh      = 0;   c_cross_mh = 0

    nodes = graph.nodes
    n     = len(nodes)

    # ── Pass 1: direct path (no ε, no graph) ──────────────────────────────────
    print("  pass 1: direct path ...")
    for source in nodes:
        for target in nodes:
            if target.color != Color.BLUE:
                continue
            if target.tir <= source.tir:
                continue
            gain  = target.tir - source.tir
            cross = source.subject != target.subject
            is_c  = (source.color == Color.RED)

            a_sh_tir.append(gain)
            if cross: a_cross_sh += 1
            for k in FEATURE_KEYS:
                a_sh_feat[k].append(
                    abs(target.features[k] - source.features[k]) / DELTA[k])
            if is_c:
                c_sh_tir.append(gain)
                if cross: c_cross_sh += 1
                for k in FEATURE_KEYS:
                    c_sh_feat[k].append(
                        abs(target.features[k] - source.features[k]) / DELTA[k])

    # ── Pass 2: multi-hop via Dijkstra ────────────────────────────────────────
    print("  pass 2: multi-hop ...")
    for i, source in enumerate(nodes):
        print(f"\r    [{i+1:3d}/{n}]  {source.subject}  TIR={source.tir:.1f}%",
              end="", flush=True)
        dist, prev = _dijkstra_all(graph, source.node_id)

        for target in nodes:
            if target.color != Color.BLUE:
                continue
            if target.tir <= source.tir:
                continue
            if target.node_id not in dist:
                continue

            path  = reconstruct_path(graph, prev, target.node_id)
            hops  = len(path) - 1
            if hops < 2:
                continue

            cross = any(nd.subject != source.subject for nd in path)
            steps = [path[j+1].tir - path[j].tir for j in range(hops)]
            pmean = float(np.mean(steps))
            is_c  = (source.color == Color.RED)

            a_mh_flat.extend(steps);       a_mh_path_means.append(pmean)
            a_mh_hops.append(hops)
            if cross: a_cross_mh += 1
            for j in range(hops):
                aa, bb = path[j], path[j+1]
                for k in FEATURE_KEYS:
                    a_mh_feat[k].append(
                        abs(bb.features[k] - aa.features[k]) / DELTA[k])
            if is_c:
                c_mh_flat.extend(steps);   c_mh_path_means.append(pmean)
                c_mh_hops.append(hops)
                if cross: c_cross_mh += 1
                for j in range(hops):
                    aa, bb = path[j], path[j+1]
                    for k in FEATURE_KEYS:
                        c_mh_feat[k].append(
                            abs(bb.features[k] - aa.features[k]) / DELTA[k])

    print()
    return {
        "a_n_sh": len(a_sh_tir),    "a_n_mh": len(a_mh_hops),
        "a_cross_sh": a_cross_sh,    "a_cross_mh": a_cross_mh,
        "a_sh_tir": a_sh_tir,        "a_mh_flat": a_mh_flat,
        "a_mh_path_means": a_mh_path_means,  "a_mh_hops": a_mh_hops,
        "a_sh_feat": a_sh_feat,      "a_mh_feat": a_mh_feat,

        "c_n_sh": len(c_sh_tir),    "c_n_mh": len(c_mh_hops),
        "c_cross_sh": c_cross_sh,    "c_cross_mh": c_cross_mh,
        "c_sh_tir": c_sh_tir,        "c_mh_flat": c_mh_flat,
        "c_mh_path_means": c_mh_path_means,  "c_mh_hops": c_mh_hops,
        "c_sh_feat": c_sh_feat,      "c_mh_feat": c_mh_feat,
    }


def compute_nearest_paths(graph):
    """
    B and D: nearest-BLUE queries.
    For each source node, independently identifies:
      target_d  nearest BLUE by raw CBTD          (direct path's natural choice)
      target_g  nearest BLUE by Dijkstra cost      (graph's natural choice)

    Then computes all 4 combinations:
      (1) direct → target_d   (2) direct → target_g
      (3) graph  → target_d   (4) graph  → target_g

    B: source = any node.   D: source = RED node (subset of B).
    """
    nodes      = graph.nodes
    blue_nodes = [nd for nd in nodes if nd.color == Color.BLUE]
    n          = len(nodes)

    def _make():
        return {
            "n": 0, "agree": 0,
            "hop_counts": defaultdict(int),
            "c1": {"tir": [], "cbtd": [], "feat": defaultdict(list)},
            "c2": {"tir": [], "cbtd": [], "feat": defaultdict(list)},
            "c3": {"steps": [], "pmeans": [], "feat": defaultdict(list),
                   "unreachable": 0, "paths": []},
            "c4": {"steps": [], "pmeans": [], "feat": defaultdict(list),
                   "paths": []},
        }

    b = _make()
    d = _make()

    for i, source in enumerate(nodes):
        print(f"\r  [{i+1:3d}/{n}]  {source.subject}  TIR={source.tir:.1f}%",
              end="", flush=True)

        valid = [bl for bl in blue_nodes if bl.tir > source.tir]
        if not valid:
            continue

        is_d = (source.color == Color.RED)

        # target_d: nearest BLUE by raw CBTD
        target_d = min(valid, key=lambda bl: _raw_cbtd(source, bl))
        cbtd_d   = _raw_cbtd(source, target_d)
        tir_d    = target_d.tir - source.tir
        feat_d   = {k: abs(target_d.features[k] - source.features[k]) / DELTA[k]
                    for k in FEATURE_KEYS}

        # Dijkstra from source
        dist, prev = _dijkstra_all(graph, source.node_id)

        # target_g: nearest BLUE by min Dijkstra path cost
        reachable = [(dist[bl.node_id], bl) for bl in valid if bl.node_id in dist]

        for result, include in [(b, True), (d, is_d)]:
            if not include:
                continue
            result["n"] += 1

            # combo 1: direct → target_d
            result["c1"]["tir"].append(tir_d)
            result["c1"]["cbtd"].append(cbtd_d)
            for k in FEATURE_KEYS:
                result["c1"]["feat"][k].append(feat_d[k])

            if not reachable:
                result["c3"]["unreachable"] += 1
                continue

            _, target_g = min(reachable)
            cbtd_g = _raw_cbtd(source, target_g)
            tir_g  = target_g.tir - source.tir
            feat_g = {k: abs(target_g.features[k] - source.features[k]) / DELTA[k]
                      for k in FEATURE_KEYS}

            if target_d.node_id == target_g.node_id:
                result["agree"] += 1

            # combo 2: direct → target_g
            result["c2"]["tir"].append(tir_g)
            result["c2"]["cbtd"].append(cbtd_g)
            for k in FEATURE_KEYS:
                result["c2"]["feat"][k].append(feat_g[k])

            # combo 4: graph → target_g
            path_g  = reconstruct_path(graph, prev, target_g.node_id)
            hops_g  = len(path_g) - 1
            result["hop_counts"][hops_g] += 1
            steps_g = [path_g[j+1].tir - path_g[j].tir for j in range(hops_g)]
            result["c4"]["steps"].extend(steps_g)
            result["c4"]["pmeans"].append(float(np.mean(steps_g)))
            result["c4"]["paths"].append(path_g)
            for j in range(hops_g):
                aa, bb = path_g[j], path_g[j+1]
                for k in FEATURE_KEYS:
                    result["c4"]["feat"][k].append(
                        abs(bb.features[k] - aa.features[k]) / DELTA[k])

            # combo 3: graph → target_d (may be unreachable)
            if target_d.node_id in dist:
                path_d  = reconstruct_path(graph, prev, target_d.node_id)
                hops_d  = len(path_d) - 1
                steps_d = [path_d[j+1].tir - path_d[j].tir for j in range(hops_d)]
                result["c3"]["steps"].extend(steps_d)
                result["c3"]["pmeans"].append(float(np.mean(steps_d)))
                result["c3"]["paths"].append(path_d)
                for j in range(hops_d):
                    aa, bb = path_d[j], path_d[j+1]
                    for k in FEATURE_KEYS:
                        result["c3"]["feat"][k].append(
                            abs(bb.features[k] - aa.features[k]) / DELTA[k])
            else:
                result["c3"]["unreachable"] += 1

    print()
    return {"b": b, "d": d}


def compute_spearman(graph):
    dists, gaps = [], []
    for a in graph.nodes:
        for b_id, w_sq in graph.edges.get(a.node_id, []):
            b = graph.node_index[b_id]
            dists.append(math.sqrt(w_sq))
            gaps.append(b.tir - a.tir)
    rho, pval = spearmanr(dists, gaps)
    return float(rho), float(pval), dists, gaps