import heapq

from node import Node
from graph import Graph


# ---------------------------------------------------------------
# Private — internal helpers, not called directly
# ---------------------------------------------------------------

def _dijkstra(graph: Graph, source_id: int, target_id: int) -> list[Node] | None:
    """
    Dijkstra's algorithm on the graph.
    Takes a source and target node ID, returns the minimum-cost path
    as an ordered list of Node objects from source to target,
    or None if target is unreachable.
    """
    INF  = float("inf")
    dist = {source_id: 0.0}
    prev = {source_id: None}
    heap = [(0.0, source_id)]

    while heap:
        cost, u = heapq.heappop(heap)

        if u == target_id:
            break

        if cost > dist.get(u, INF):
            continue

        for v, w in graph.edges.get(u, []):
            new_cost = cost + w
            if new_cost < dist.get(v, INF):
                dist[v] = new_cost
                prev[v] = u
                heapq.heappush(heap, (new_cost, v))

    if target_id not in dist:
        return None

    path, current = [], target_id

    while current is not None:
        path.append(graph.node_index[current])
        current = prev[current]

    path.reverse()
    return path


def _dijkstra_all(graph: Graph, source_id: int) -> tuple[dict, dict]:
    """
    Dijkstra from source to ALL reachable nodes.
    Returns (dist dict, prev dict) for path reconstruction.
    """
    INF = float("inf")
    dist = {source_id: 0.0}
    prev = {source_id: None}
    heap = [(0.0, source_id)]

    while heap:
        cost, u = heapq.heappop(heap)

        if cost > dist.get(u, INF):
            continue

        for v, w in graph.edges.get(u, []):
            new_cost = cost + w

            if new_cost < dist.get(v, INF):
                dist[v] = new_cost
                prev[v] = u
                heapq.heappush(heap, (new_cost, v))

    return dist, prev


# ---------------------------------------------------------------
# Public query functions
# ---------------------------------------------------------------

def find_path(graph: Graph, source_id: int, target_id: int) -> list[Node] | None:
    """
    Find the minimum-cost path between two nodes already in the graph.
    Returns an ordered list of Node objects, or None if unreachable.
    """
    return _dijkstra(graph, source_id, target_id)


def find_path_from_query(graph: Graph, query_node: Node, target_id: int) -> list[Node] | None:
    """
    Find the minimum-cost path from a virtual node (not part of the graph)
    to a target node already in the graph.
    Uses the same clinical epsilon as the graph itself.
    Graph is left unchanged after the call.
    """
    distances = {node.node_id: graph.distance(query_node, node) for node in graph.nodes}

    virtual_edges = [
        (node.node_id, distances[node.node_id])
        for node in graph.nodes
        if distances[node.node_id] <= graph.epsilon and node.tir > query_node.tir
    ]

    if not virtual_edges:
        print("Query node has no reachable neighbours at the clinical epsilon.")
        return None

    graph.nodes.append(query_node)
    graph.node_index[query_node.node_id] = query_node
    graph.edges[query_node.node_id] = virtual_edges

    path = _dijkstra(graph, query_node.node_id, target_id)

    graph.nodes.remove(query_node)
    del graph.node_index[query_node.node_id]
    del graph.edges[query_node.node_id]

    return path


def find_path_to_nearest_blue(graph: Graph, source_id: int) -> list[Node] | None:
    """
    Find the minimum-cost path from source to the nearest reachable blue node.
    Target is whichever blue node Dijkstra reaches with the lowest total cost.
    """
    from node import Color
    dist, prev = _dijkstra_all(graph, source_id)

    candidates = [
        (dist[n.node_id], n.node_id)
        for n in graph.nodes
        if n.color == Color.BLUE and n.node_id in dist
    ]

    if not candidates:
        return None

    _, best_id = min(candidates)

    path, current = [],

    while current is not None:
        path.append(graph.node_index[current])
        current = prev[current]

    path.reverse()

    return path