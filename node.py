from enum import Enum

import numpy as np
import pandas as pd

from config import FEATURE_KEYS, TIR_THRESHOLD

class Color(Enum):
    RED  = "red"
    BLUE = "blue"


class Node:
    def __init__(self, node_id: int, row: pd.Series):
        self.node_id = node_id + 1
        self.subject = row["subject"]
        self.date = row["date"]
        self.tir = round(float(row["tir"]), 2)

        self.features = {k: round(float(row[k]), 2) for k in FEATURE_KEYS}

        self.color = Color.BLUE if self.tir >= TIR_THRESHOLD else Color.RED

    @property
    def feature_vector(self) -> np.ndarray:
        return np.array([self.features[k] for k in FEATURE_KEYS], dtype=float)

    def __repr__(self):
        return (
            f"Node(id={self.node_id}, subject={self.subject}, "
            f"date={self.date}, tir={self.tir:.1f}%, color={self.color})"
        )