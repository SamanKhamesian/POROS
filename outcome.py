"""
outcome.py — Outcome measured along a returned path (Algorithm 2).

The graph is always built on observed TIR (Eq. adjacency). This module only
decides which TIR is REPORTED along a path once it has been found, switched by
config.OUTCOME:

  "observed"  : the TIR observed at each node of the path
  "predicted" : the source's observed TIR, then the TIR estimated at each step
                by the predictor of the individual p to whom the source belongs
                ŷ_j = Ĥ_p(v_i, v_j − v_i, ŷ_i),  ŷ_0 = H(x)

Predictor
  One Ridge regressor per individual, trained on ALL pairs of consecutive
  observations of that individual (no held-out split). Inputs, scaling and
  hyperparameters are identical to evaluation/eval_predictor.py.

Usage
  graph.outcome = Outcome(graph.nodes)
  tirs = graph.outcome.path(path)              # list of TIR, one per node
  tir  = graph.outcome.direct(source, target)  # TIR at target after one jump

  Both return None in predicted mode when the source's individual has no
  model (fewer than MIN_PAIRS consecutive pairs).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from config import FEATURE_KEYS, OUTCOME
from evaluation.eval_predictor import MODEL_CONFIG, build_transitions

PREDICTOR = "Ridge Regression"
MIN_PAIRS = 2   # same minimum as evaluation/eval_predictor.py


class Outcome:
    def __init__(self, nodes):
        if OUTCOME not in ("observed", "predicted"):
            raise ValueError(f"config.OUTCOME must be 'observed' or 'predicted', got {OUTCOME!r}")

        self.mode = OUTCOME
        self.models = {}   # subject → (w, b), scaler folded into the coefficients

        if self.mode == "predicted":
            self._fit(nodes)

        subjects = {nd.subject for nd in nodes}
        if self.mode == "observed":
            print(f"  outcome: observed TIR")
        else:
            print(f"  outcome: predicted TIR ({PREDICTOR}, "
                  f"{len(self.models)}/{len(subjects)} subjects with a model)")

    # ------------------------------------------------------------------

    def _fit(self, nodes):
        df = pd.DataFrame([{"subject": nd.subject, "date": nd.date, "tir": nd.tir,
                            **nd.features} for nd in nodes])

        for subject, subject_df in df.groupby("subject"):
            X, y, _ = build_transitions(subject_df)
            if len(y) < MIN_PAIRS:
                continue

            scaler = StandardScaler().fit(X)
            model = Ridge(**MODEL_CONFIG[PREDICTOR]["params"]).fit(scaler.transform(X), y)

            # ŷ = ((x − μ) / σ) · coef + intercept  =  x · w + b
            w = model.coef_ / scaler.scale_
            b = float(model.intercept_ - np.dot(scaler.mean_ / scaler.scale_, model.coef_))
            self.models[subject] = (w, b)

    def _step(self, subject, node_i, node_j, tir_i):
        w, b = self.models[subject]
        v_i, v_j = node_i.feature_vector, node_j.feature_vector
        x = np.concatenate([v_i, v_j - v_i, [tir_i]])
        return float(np.dot(x, w) + b)

    # ------------------------------------------------------------------

    def has_model(self, source):
        return self.mode == "observed" or source.subject in self.models

    def path(self, path):
        """TIR at every node of the path, source first."""
        if self.mode == "observed":
            return [nd.tir for nd in path]

        source = path[0]
        if source.subject not in self.models:
            return None

        tirs = [source.tir]
        for j in range(len(path) - 1):
            tirs.append(self._step(source.subject, path[j], path[j + 1], tirs[-1]))
        return tirs

    def direct(self, source, target):
        """TIR at the target after a single direct jump from the source."""
        if self.mode == "observed":
            return target.tir

        if source.subject not in self.models:
            return None
        return self._step(source.subject, source, target, source.tir)
