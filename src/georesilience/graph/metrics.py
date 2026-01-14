from __future__ import annotations

from dataclasses import dataclass

import networkx as nx


@dataclass(frozen=True)
class NodeMetrics:
    betweenness: dict[str, float]


def compute_node_metrics(graph: nx.Graph) -> NodeMetrics:
    bet = nx.betweenness_centrality(graph, normalized=True)
    # Ensure keys are str (our node ids are strings)
    bet_str: dict[str, float] = {str(k): float(v) for k, v in bet.items()}
    return NodeMetrics(betweenness=bet_str)
