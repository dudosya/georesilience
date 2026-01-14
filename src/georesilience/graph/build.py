from __future__ import annotations

import networkx as nx
import polars as pl


def build_graph_from_parquet(*, nodes: pl.DataFrame, edges: pl.DataFrame) -> nx.Graph:
    g = nx.Graph()

    for row in nodes.iter_rows(named=True):
        g.add_node(row["node_id"], kind=row.get("kind"), x=row.get("x"), y=row.get("y"))

    for row in edges.iter_rows(named=True):
        g.add_edge(
            row["u"],
            row["v"],
            edge_id=row.get("edge_id"),
            kind=row.get("kind"),
            length_m=row.get("length_m"),
        )

    return g
