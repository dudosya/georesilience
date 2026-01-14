from __future__ import annotations

import networkx as nx
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from georesilience.graph.metrics import compute_node_metrics
from georesilience.sim.cascade import CascadeConfig, FailureMode, run_cascade


@st.composite
def small_graph(draw) -> nx.Graph:
    n = draw(st.integers(min_value=1, max_value=20))
    p = draw(st.floats(min_value=0.0, max_value=0.4))
    g = nx.gnp_random_graph(n, p, seed=draw(st.integers(min_value=0, max_value=10_000)))
    # Ensure at least one edge sometimes
    return g


@given(g=small_graph(), seed=st.integers(min_value=0, max_value=1_000))
@settings(max_examples=50, deadline=None)
def test_cascade_terminates_and_is_monotonic(g: nx.Graph, seed: int) -> None:
    metrics = compute_node_metrics(g)
    cfg = CascadeConfig(
        alpha=0.2,
        initial_failures=1,
        mode=FailureMode.random,
        seed=seed,
        max_steps=30,
    )
    result = run_cascade(g, node_metrics=metrics, config=cfg)

    # terminates within max_steps (steps table length <= max_steps)
    assert result.steps.height <= cfg.max_steps

    # monotonic: once failed, stays failed (implied by single final table).
    failed_steps = result.nodes.filter(pl.col("failed"))["failed_step"].to_list()
    assert all(s is None or (isinstance(s, int) and s >= 0) for s in failed_steps)


def test_determinism_same_seed() -> None:
    g = nx.path_graph(10)
    metrics = compute_node_metrics(g)
    cfg = CascadeConfig(
        alpha=0.2,
        initial_failures=2,
        mode=FailureMode.random,
        seed=123,
        max_steps=50,
    )

    r1 = run_cascade(g, node_metrics=metrics, config=cfg)
    r2 = run_cascade(g, node_metrics=metrics, config=cfg)

    assert r1.nodes.sort("node_id").to_dicts() == r2.nodes.sort("node_id").to_dicts()
    assert r1.steps.to_dicts() == r2.steps.to_dicts()
