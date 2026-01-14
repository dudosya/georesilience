from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import polars as pl


@dataclass(frozen=True)
class NodeMetrics:
    betweenness: dict[str, float]
    degree: dict[str, float]
    closeness: dict[str, float]
    component_id: dict[str, int]
    component_size: dict[str, int]
    eigenvector: dict[str, float] | None


def compute_node_metrics(graph: nx.Graph, *, advanced: bool = False) -> NodeMetrics:
    bet = nx.betweenness_centrality(graph, normalized=True)
    deg = dict(graph.degree())
    clo = nx.closeness_centrality(graph)

    comp_id: dict[str, int] = {}
    comp_size: dict[str, int] = {}
    for idx, component in enumerate(nx.connected_components(graph)):
        size = len(component)
        for n in component:
            n_str = str(n)
            comp_id[n_str] = idx
            comp_size[n_str] = size

    eig: dict[str, float] | None = None
    if advanced and graph.number_of_nodes() > 0:
        # Can be slow on big graphs; gated behind --advanced-metrics.
        try:
            eig_raw = nx.eigenvector_centrality(graph, max_iter=2000)
            eig = {str(k): float(v) for k, v in eig_raw.items()}
        except Exception:
            eig = None

    return NodeMetrics(
        betweenness={str(k): float(v) for k, v in bet.items()},
        degree={str(k): float(v) for k, v in deg.items()},
        closeness={str(k): float(v) for k, v in clo.items()},
        component_id=comp_id,
        component_size=comp_size,
        eigenvector=eig,
    )


def node_metrics_frame(metrics: NodeMetrics) -> pl.DataFrame:
    keys = set(metrics.betweenness) | set(metrics.degree) | set(metrics.closeness)
    keys |= set(metrics.component_id) | set(metrics.component_size)
    if metrics.eigenvector is not None:
        keys |= set(metrics.eigenvector)

    rows: list[dict[str, object]] = []
    for node_id in sorted(keys):
        row: dict[str, object] = {
            "node_id": node_id,
            "degree": float(metrics.degree.get(node_id, 0.0)),
            "closeness": float(metrics.closeness.get(node_id, 0.0)),
            "betweenness": float(metrics.betweenness.get(node_id, 0.0)),
            "component_id": int(metrics.component_id.get(node_id, -1)),
            "component_size": int(metrics.component_size.get(node_id, 0)),
        }
        if metrics.eigenvector is not None:
            row["eigenvector"] = float(metrics.eigenvector.get(node_id, 0.0))
        rows.append(row)
    return pl.from_dicts(rows)


def edge_metrics_frame(graph: nx.Graph, *, advanced: bool = False) -> pl.DataFrame:
    # Baseline: edge betweenness + bridge detection
    ebet = nx.edge_betweenness_centrality(graph, normalized=True)
    bridges = {tuple(sorted((str(u), str(v)))) for (u, v) in nx.bridges(graph)}

    rows: list[dict[str, object]] = []
    for (u, v), val in ebet.items():
        u_str = str(u)
        v_str = str(v)
        key = tuple(sorted((u_str, v_str)))
        rows.append(
            {
                "u": u_str,
                "v": v_str,
                "edge_betweenness": float(val),
                "is_bridge": key in bridges,
            }
        )

    # Placeholder for future advanced edge metrics
    _ = advanced
    if rows:
        return pl.from_dicts(rows)
    return pl.DataFrame({"u": [], "v": [], "edge_betweenness": [], "is_bridge": []})
