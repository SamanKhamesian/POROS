"""
face.py — FACE baseline (Poyiadzi et al., 2020) re-instantiated on the
          Behavioral Progression setting, for comparison against POROS.

Purpose
───────────────────────────────────────────────────────────────────────────────
POROS is a path-aware counterfactual method, so its natural comparison point is
the path-aware CE line of work — chiefly FACE. This file re-instantiates FACE on
our patient-day data so that, in a later evaluation file, we can contrast the
path FACE produces against the path POROS produces to the SAME target.

The whole point of the comparison is a STRUCTURAL one, so the differences between
FACE and POROS must be preserved faithfully and NOT accidentally papered over:

  POROS edge  :  DIRECTED,   admitted iff  d ≤ ε  AND  TIR(j) > TIR(i)
                 weight = d²           (super-linear → prefers gradual multi-hop)
                 ⇒ every path is monotone-improving in TIR, by construction.

  FACE edge   :  UNDIRECTED, admitted iff the two nodes are "close"
                 weight = -log( p̂(midpoint) ) · d   (density-weighted)
                 ⇒ NO outcome condition on edges. A FACE path may step BACKWARD
                   in TIR (through RED intermediates) on its way to a BLUE target.
                 This is precisely the contrast we want to measure.

What we keep faithful to the original FACE (Poyiadzi et al., 2020, Alg. 1)
───────────────────────────────────────────────────────────────────────────────
  • Undirected graph over real observed instances (no synthetic points).

  • Density-weighted edges:  w_ij = w(p̂(midpoint)) · d(x_i, x_j),  w(z) = -log(z),
    estimated with a Kernel Density Estimator (their Eq. 2 / experimental setting).

  • Target = an instance of the DESIRED CLASS selected by minimum density-weighted
    shortest-path cost (their candidate-target set I_CT + Dijkstra, Alg. 1 line 11).

  • Optional density threshold t_d on candidate targets (their p̂(x) ≥ t_d).

What we ADAPT to our model-free, continuous-outcome setting (documented on purpose)
───────────────────────────────────────────────────────────────────────────────
  • No classifier. FACE's confidence gate  clf(x) ≥ t_p  is replaced by the
    clinical class label:  candidate targets are BLUE nodes (TIR ≥ τ).
    
  • Distance function = CBTD (our metric), NOT FACE's l2. This is a deliberate
    control: holding the metric fixed across both methods isolates the mechanism
    (undirected + density routing vs. directed + d²) rather than confounding it
    with a change of metric. Using CBTD if anything strengthens FACE.

  • Connectivity = k-NN graph (their Eq. 3 construction) by default, so that the
    FACE graph carries NO dependence on POROS's principled ε — ε is POROS's own
    contribution and must not leak into the baseline. An ε-graph mode is available
    but, if used, takes a FACE-SPECIFIC threshold passed explicitly by the caller;
    it must never be graph's epsilon.

The KDE is fit in CBTD-normalized coordinate space  (coord_k = sqrt(w_k)·v_k/δ_k),
so that Euclidean geometry in that space reproduces CBTD exactly — density and
distance then live in the same commensurable space.

Functions / classes
  FaceGraph                 : undirected, density-weighted k-NN (or ε) graph
  face_shortest_paths       : Dijkstra from a source over the undirected graph
  reconstruct_path          : node-list path from a Dijkstra prev-map
  find_face_counterfactual  : FACE's output — target (BLUE, min-cost) + its path
"""

import heapq
from collections import defaultdict

import numpy as np
from scipy.stats import gaussian_kde

from config import DATASET_FOLDER, FEATURE_KEYS, DELTA
from distance import DistanceMetric
from graph import create_nodes
from node import Node, Color


_TINY = 1e-300  # floor on density before -log, guards log(0)


# ── FACE graph ────────────────────────────────────────────────────────────────

class FaceGraph:
    """
    Undirected, density-weighted graph re-instantiating FACE on patient-days.

    Parameters
    ----------
    nodes         : list[Node]
    metric        : DistanceMetric | None
        Reuse POROS's metric to guarantee an identical CBTD (no recompute). If
        None, a fresh DistanceMetric with uniform weights is built (matches Graph).
    connectivity  : {"knn", "epsilon"}
        "knn"     — connect each node to its k nearest by CBTD (FACE Eq. 3 style);
                    carries NO dependence on POROS's ε. This is the default.
        "epsilon" — connect pairs with CBTD ≤ face_epsilon. face_epsilon is a
                    FACE-SPECIFIC threshold and MUST NOT be POROS's graph.epsilon.
    k             : int    — neighbors per node in k-NN mode.
    face_epsilon  : float  — distance cutoff in epsilon mode (required if used).
    kde_bw        : str|float — bandwidth passed to scipy gaussian_kde.

    Attributes
    ----------
    edges     : dict  node_id → [(neighbor_id, weight), ...]   (symmetric)
    density   : dict  node_id → p̂(node)   (for optional t_d target filtering)
    """

    def __init__(self, nodes, metric=None, connectivity="knn",
                 k=10, face_epsilon=None, kde_bw="scott"):
        if connectivity == "epsilon" and face_epsilon is None:
            raise ValueError("epsilon mode requires an explicit FACE-specific "
                             "face_epsilon (never pass POROS's graph.epsilon).")

        self.nodes        = nodes
        self.node_index   = {n.node_id: n for n in nodes}
        self.metric       = metric if metric is not None else DistanceMetric(nodes, True)
        self.connectivity = connectivity
        self.k            = k
        self.face_epsilon = face_epsilon
        self.edges        = defaultdict(list)

        # CBTD-normalized coordinates: Euclidean here == CBTD.
        w = self.metric.weights
        scale = np.array([np.sqrt(w[key]) / DELTA[key] for key in FEATURE_KEYS])
        self._coords = np.array(
            [[nd.features[key] for key in FEATURE_KEYS] for nd in nodes]
        ) * scale                                   # shape (N, K)
        self._id_row = {nd.node_id: i for i, nd in enumerate(nodes)}

        # KDE over all nodes in the same (CBTD) space FACE routes through.
        self._kde = gaussian_kde(self._coords.T, bw_method=kde_bw)
        dens_nodes = self._kde(self._coords.T)                       # p̂ at each node
        self.density = {nd.node_id: float(dens_nodes[i])
                        for i, nd in enumerate(nodes)}

        self.build_edges()

    # ------------------------------------------------------------------

    def _candidate_pairs(self):
        """Undirected pairs (i<j) admitted by the chosen connectivity rule."""
        ids = [nd.node_id for nd in self.nodes]
        pair = set()

        if self.connectivity == "knn":
            for a in ids:
                nbrs = sorted(
                    (b for b in ids if b != a),
                    key=lambda b: self.metric.pairwise[(a, b)]
                )[:self.k]
                for b in nbrs:                       # symmetrize by union
                    pair.add((min(a, b), max(a, b)))
        else:  # epsilon
            for idx, a in enumerate(ids):
                for b in ids[idx + 1:]:
                    if self.metric.pairwise[(a, b)] <= self.face_epsilon:
                        pair.add((a, b))
        return list(pair)

    def build_edges(self):
        """
        Admit undirected edges with FACE density weights:
            w_ij = -log( p̂( midpoint_ij ) ) · d(x_i, x_j).

        Midpoint densities are evaluated in one batched KDE call. Weights are
        floored at 0 so Dijkstra stays valid (p̂ > 1 → -log < 0 can occur only
        for extremely dense midpoints; in the high-dimensional CBTD space this is
        rare, and a floor of 0 keeps such edges "cheap", consistent with FACE).
        """
        self.edges.clear()
        pairs = self._candidate_pairs()
        if not pairs:
            return

        mids = np.array([
            (self._coords[self._id_row[a]] + self._coords[self._id_row[b]]) / 2.0
            for a, b in pairs
        ])                                                   # (P, K)
        dens = self._kde(mids.T)                             # (P,)

        n_clamped = 0
        for (a, b), pd in zip(pairs, dens):
            d = self.metric.pairwise[(a, b)]
            raw = -np.log(max(pd, _TINY))
            if raw < 0.0:
                raw = 0.0
                n_clamped += 1
            w = float(raw) * d
            self.edges[a].append((b, w))
            self.edges[b].append((a, w))                     # undirected

        if n_clamped:
            print(f"  [FACE] {n_clamped}/{len(pairs)} edges had p̂>1 at midpoint; "
                  f"weight floored at 0.")

    # ------------------------------------------------------------------

    def summary(self):
        deg = [len(v) for v in self.edges.values()]
        n_edges = sum(deg) // 2
        mode = (f"k-NN (k={self.k})" if self.connectivity == "knn"
                else f"ε-graph (face_ε={self.face_epsilon})")
        print(f"FACE graph [{mode}] : {len(self.nodes)} nodes, {n_edges} undirected "
              f"edges, mean degree {np.mean(deg):.1f}")


# ── Search over the FACE graph ──────────────────────────────────────────────────

def face_shortest_paths(fg: FaceGraph, source_id: int):
    """Dijkstra from source over the undirected, density-weighted FACE graph."""
    INF  = float("inf")
    dist = {source_id: 0.0}
    prev = {source_id: None}
    heap = [(0.0, source_id)]

    while heap:
        cost, u = heapq.heappop(heap)
        if cost > dist.get(u, INF):
            continue
        for v, w in fg.edges.get(u, []):
            new_cost = cost + w
            if new_cost < dist.get(v, INF):
                dist[v] = new_cost
                prev[v] = u
                heapq.heappush(heap, (new_cost, v))
    return dist, prev


def reconstruct_path(fg: FaceGraph, prev: dict, target_id: int) -> list[Node]:
    path, cur = [], target_id
    while cur is not None:
        path.append(fg.node_index[cur])
        cur = prev.get(cur)
    path.reverse()
    return path


def find_face_counterfactual(fg: FaceGraph, source_id: int, density_threshold=None):
    """
    FACE's output for one source: the BLUE node reachable by the minimum
    density-weighted path cost, and that path.

    This mirrors FACE Alg. 1: candidate targets are the desired-class instances
    (BLUE here), optionally filtered by a density floor t_d, and the counterfactual
    is the reachable candidate with the smallest shortest-path cost.

    Returns
    -------
    dict | None
        {"target": Node, "path": list[Node], "cost": float, "hops": int}
        or None if no BLUE candidate is reachable from the source.
    """
    dist, prev = face_shortest_paths(fg, source_id)

    candidates = [
        (dist[nd.node_id], nd.node_id)
        for nd in fg.nodes
        if nd.color == Color.BLUE
        and nd.node_id in dist
        and nd.node_id != source_id
        and (density_threshold is None or fg.density[nd.node_id] >= density_threshold)
    ]
    if not candidates:
        return None

    cost, target_id = min(candidates)
    path = reconstruct_path(fg, prev, target_id)
    return {"target": fg.node_index[target_id],
            "path": path,
            "cost": float(cost),
            "hops": len(path) - 1}


# ── Standalone sanity check ─────────────────────────────────────────────────────

def run():
    nodes = create_nodes(DATASET_FOLDER)
    fg = FaceGraph(nodes, connectivity="knn", k=10)
    fg.summary()

    # Demonstrate on one RED source: show FACE can step backward in TIR.
    red = next((n for n in nodes if n.color == Color.RED), None)
    if red is None:
        return
    cf = find_face_counterfactual(fg, red.node_id)
    if cf is None:
        print(f"  RED source {red.node_id}: no reachable BLUE candidate.")
        return

    tirs = [nd.tir for nd in cf["path"]]
    backward = sum(1 for i in range(len(tirs) - 1) if tirs[i + 1] < tirs[i])
    print(f"\n  RED source {red.node_id} (TIR={red.tir:.1f}%) → "
          f"FACE target {cf['target'].node_id} (TIR={cf['target'].tir:.1f}%)")
    print(f"  hops={cf['hops']}, cost={cf['cost']:.2f}, "
          f"backward-TIR steps={backward}")
    print("  TIR along path: " + " → ".join(f"{t:.0f}" for t in tirs))


if __name__ == "__main__":
    run()