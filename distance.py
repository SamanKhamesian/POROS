import numpy as np
from scipy.stats import spearmanr

from node import Node
from config import FEATURE_KEYS, DELTA


class DistanceMetric:
    """
    Clinical Behavioral Transition Distance (CBTD).

    d(a, b) = sqrt( sum_k  w_k * ((a_k - b_k) / delta_k)^2 )

    delta_k  — minimum clinically meaningful change for feature k (MCID).
    w_k      — Spearman |rho| between feature k and TIR, normalized so
               weights average to 1 across all features.  Features more
               strongly correlated with TIR contribute more to the distance;
               normalization keeps distances in interpretable MCID units
               regardless of the cohort-level correlation magnitudes.

    Initializing with the full node list precomputes weights and all
    pairwise distances.  Graph and analysis utilities read from self.pairwise
    rather than recomputing.
    """

    def __init__(self, nodes: list[Node], uniform_weights: bool = False):
        self.weights = self._compute_weights(nodes, uniform=uniform_weights)
        self.pairwise = self._precompute_pairwise(nodes)

    # ------------------------------------------------------------------

    @staticmethod
    def _compute_weights(nodes: list[Node], uniform: bool = False) -> dict:
        """
        w_k = |Spearman(feature_k, TIR)|, normalized so mean(w_k) = 1.

        Raw Spearman correlations are often all below 0.5 on small cohorts,
        which would globally compress distances by sqrt(mean_rho) without
        normalization.  Dividing by the mean preserves relative weighting
        (TIR-correlated features still matter more) while keeping the metric
        in MCID units.
        A floor of 1e-3 before normalization prevents zero weights.
        """
        if uniform:
            print("\n── Feature weights (uniform w=1) ──")
            return {k: 1.0 for k in FEATURE_KEYS}


        tir_vals = np.array([n.tir for n in nodes])
        raw = {}

        for k in FEATURE_KEYS:
            feat_vals = np.array([n.features[k] for n in nodes])
            rho, _ = spearmanr(feat_vals, tir_vals)
            raw[k] = max(abs(float(rho)), 1e-3)

        mean_w = float(np.mean(list(raw.values())))
        weights = {k: v / mean_w for k, v in raw.items()}

        print("\n── Feature weights (normalised |Spearman ρ|) ──")
        for k, w in weights.items():
            rho = raw[k]
            print(f"  {k:<35}  ρ = {rho:+.3f}   w = {w:.3f}")

        return weights

    def _precompute_pairwise(self, nodes: list[Node]) -> dict:
        """
        Compute CBTD for every ordered pair of nodes once at init.
        Stored in self.pairwise[(id_a, id_b)] for O(1) lookup.
        """
        distances = {}
        for node_a in nodes:
            for node_b in nodes:
                if node_a.node_id == node_b.node_id:
                    continue
                distances[(node_a.node_id, node_b.node_id)] = self.compute(node_a, node_b)
        return distances


    def compute_epsilon(self, nodes: list[Node]) -> float:
        """
        Derive epsilon from within-subject consecutive-day transitions.

        Algorithm
        ---------
        1. For each subject, sort their patient-days chronologically and compute
           the CBTD between every consecutive pair of days.
        2. Take the MAX of those distances per subject — the largest single-day
           behavioral change this subject has ever demonstrated.
        3. Take the MIN across all subjects — the behavioral step size that
           every patient in the cohort has proven they can make.

        Guarantee
        ---------
        Every edge in the graph (including cross-patient edges) has d <= epsilon.
        Since every patient has demonstrated a behavioral change of at least
        epsilon within their own history, every recommended step in any path
        is guaranteed to be within the patient's demonstrated capability.
        """
        from collections import defaultdict

        subject_nodes = defaultdict(list)
        for node in nodes:
            subject_nodes[node.subject].append(node)

        per_subject_max = {}

        for subject, s_nodes in subject_nodes.items():
            sorted_nodes = sorted(s_nodes, key=lambda n: n.date)

            if len(sorted_nodes) < 2:
                continue

            consec = [self.pairwise[(sorted_nodes[i].node_id, sorted_nodes[i + 1].node_id)] for i in range(len(sorted_nodes) - 1)]

            per_subject_max[subject] = float(np.max(consec))

        epsilon = float(np.min(list(per_subject_max.values())))
        limiting = min(per_subject_max, key=per_subject_max.get)

        print("\n── Per-subject max consecutive-day CBTD ──")
        for subj, max_d in sorted(per_subject_max.items()):
            marker = "  ← limiting" if subj == limiting else ""
            print(f"  {subj:<20}  {max_d:.4f}{marker}")
        print(f"\n  ε = {epsilon:.4f}  (min of per-subject maxima, {len(per_subject_max)} subjects)")

        return epsilon

    # ------------------------------------------------------------------

    def compute(self, a: Node, b: Node) -> float:
        """Return the CBTD between two patient-day nodes."""
        va = a.feature_vector
        vb = b.feature_vector

        delta = np.array([DELTA[k] for k in FEATURE_KEYS])
        w = np.array([self.weights[k] for k in FEATURE_KEYS])

        diff = (va - vb) / delta

        return float(np.sqrt(np.sum(w * diff ** 2)))