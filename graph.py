from collections import defaultdict

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import TwoSlopeNorm

from config import DATASET, DATASET_FOLDER
from distance import DistanceMetric
from node import Node

if DATASET == "ExActHealth":
    from preprocess import build_dataset as _build_fn
else:
    from preprocess_uom import build_dataset_uom as _build_fn


def create_nodes(folder_path):
    database = _build_fn(folder_path)
    nodes = []

    for index, day in database.iterrows():
        new_node = Node(node_id=index, row=day)
        nodes.append(new_node)

    return nodes


# ── Graph ─────────────────────────────────────────────────────────────────────

class Graph:
    def __init__(self, nodes: list[Node]):
        self.nodes      = nodes
        self.node_index = {n.node_id: n for n in nodes}
        self.edges      = defaultdict(list)
        self.metric     = DistanceMetric(nodes, True)
        self.epsilon    = self.metric.compute_epsilon_2(nodes)
        self.build_edges(self.epsilon)

    # ------------------------------------------------------------------

    def distance(self, a: Node, b: Node) -> float:
        """Delegate to DistanceMetric. Supports virtual query nodes not in pairwise cache."""
        key = (a.node_id, b.node_id)

        if key in self.metric.pairwise:
            return self.metric.pairwise[key]

        return self.metric.compute(a, b)

    def build_edges(self, epsilon: float) -> None:
        """
        Add a directed edge A -> B if and only if:
          1. distance(A, B) <= epsilon   (behavioral proximity)
          2. TIR(B) > TIR(A)             (monotone improvement)
        """
        self.edges.clear()

        for node_a in self.nodes:
            for node_b in self.nodes:

                if node_a.node_id == node_b.node_id:
                    continue

                if node_b.tir <= node_a.tir:
                    continue

                d = self.metric.pairwise[(node_a.node_id, node_b.node_id)]

                if d <= epsilon:
                    self.edges[node_a.node_id].append((node_b.node_id, d**2))

    def find_node(self, subject: str, date: str) -> Node | None:
        """Look up a node by subject name and date string (e.g. '2025-05-27')."""
        for node in self.nodes:
            if node.subject == subject and str(node.date) == date:
                return node
        print(f"No node found for {subject} on {date}.")
        return None

    def get_neighbors(self, node_id: int) -> None:
        """Print all neighbors of a given node, sorted by distance."""
        if node_id not in self.node_index:
            print(f"Node {node_id} not found in graph.")
            return

        node = self.node_index[node_id]
        neighbors = sorted(self.edges.get(node_id, []), key=lambda x: x[1])

        print(f"\nNeighbors of {node}:")

        if not neighbors:
            print("  No neighbors found.")
            return

        for neighbor_id, dist in neighbors:
            neighbor = self.node_index[neighbor_id]
            print(f"  d={dist:.3f}  →  {neighbor}")

    # ------------------------------------------------------------------

    def print_graph(self, max_neighbors: int = 5) -> None:
        """Print each node with its closest neighbors."""
        total_edges = sum(len(v) for v in self.edges.values())
        print(f"Graph: {len(self.nodes)} nodes, {total_edges} edges\n")

        for node in self.nodes:
            neighbors = sorted(self.edges.get(node.node_id, []), key=lambda x: x[1])
            neighbors = neighbors[:max_neighbors]
            neighbor_str = ", ".join(f"Node {nid} (d={self.metric.pairwise[(node.node_id, nid)]:.3f}, w={w:.3f})" for nid, w in neighbors)
            print(f"{node}  ->  [{neighbor_str}]")

    def visualize(self) -> None:
        """Spring-layout graph. Color encodes TIR continuously: red (low) -> blue (high), centered at 70%."""
        G = nx.DiGraph()

        for node in self.nodes:
            G.add_node(node.node_id)

        for a_id, neighbors in self.edges.items():
            for b_id, w in neighbors:
                G.add_edge(a_id, b_id, weight=w)

        pos = nx.spring_layout(G, seed=42, k=2)

        tir_values = [self.node_index[n].tir for n in G.nodes()]
        norm = TwoSlopeNorm(vmin=40, vcenter=70, vmax=100)
        cmap = cm.RdBu
        node_colors = [cmap(norm(t)) for t in tir_values]

        plt.figure(figsize=(12, 8))

        nx.draw_networkx_edges(G, pos, edge_color="#888888", arrows=True, arrowsize=5, width=0.5, alpha=0.5)
        nx.draw_networkx_nodes(G, pos, nodelist=list(G.nodes()), node_color=node_colors, node_size=50, alpha=0.95)

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.6, pad=0.005)
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label("Time in Range (%)", fontsize=16)

        plt.title(f"Constrained-Graph for Daily Behavioral Profile\n{len(self.nodes)} nodes, {G.number_of_edges()} edges", fontsize=18)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("results/graph.png", dpi=300)
        plt.close()


def run():
    nodes = create_nodes(DATASET_FOLDER)
    graph = Graph(nodes)
    graph.print_graph()
    graph.visualize()


if __name__ == "__main__":
    run()