# GeoResilience

GeoResilience builds a lightweight geospatial “digital twin” of a city-scale power network from OpenStreetMap, converts it into a topological graph, simulates cascading failures, and trains a graph neural network to predict node-level failure risk.

Focus is on data provenance, explicit modeling assumptions, reproducibility, and inspectable artifacts (Parquet + JSON).

## Geospatial Model

### Data Source and Semantics

Ingestion uses OpenStreetMap via OSMnx. Linear infrastructure is represented by LineString features tagged as power lines/cables. Substations are represented by point or polygon features; polygons are reduced to centroids.

This is not a utility-grade network model. It is a topology-first approximation suitable for comparative resilience analysis and prototyping.

### Topology Construction

The graph is an undirected NetworkX graph. Nodes represent line endpoints and substations. Edges represent observed power line/cable segments.

Node identifiers are stable string IDs derived from coordinates (snapped to a fixed precision), enabling consistent joins across parquet artifacts.

### Geometry and Distance

Edges store WKT geometry when available.

Current note: the stored `length_m` is computed from the raw geometry length. If the input geometry is unprojected WGS84, this value is not true meters. If you need metric lengths, add a projection step before measuring.

## Resilience Engine

### Bottlenecks

The CLI reports bottlenecks using node betweenness centrality, degree, and connected-component membership/size. It also reports edge betweenness centrality and bridge detection.

The simulation command prints top-N bottleneck nodes/edges as Rich tables.

### Cascading Failure Simulation

The cascade model is Motter–Lai style, using betweenness centrality as a proxy for load.

Initial failures are selected either as `targeted` (highest base load) or `random` (seeded). Capacity per node is $C_i = (1 + \alpha) \cdot L_i$, where $L_i$ is base load. Each step recomputes load on the remaining graph; nodes exceeding capacity fail; iteration continues until stable or `max_steps`.

This is a stylized redistribution model intended for experimentation and feature generation, not a full power-flow solver.

## Predictive Risk Layer (GCN)

Training uses a lightweight GCN (Torch Geometric + PyTorch Lightning) to predict node-level failure probability.

Inputs are topological features exported from the resilience engine plus spatial coordinates (x/y) from the dataset. Labels are derived from the cascade simulation outputs. The CLI seeds Python/NumPy/PyTorch and Lightning to ensure deterministic behavior where supported.

## Artifacts

| Location                                                   | Purpose                                                                                 |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `data/<dataset>/nodes.parquet`                             | Node table (`node_id`, `kind`, `x`, `y`).                                               |
| `data/<dataset>/edges.parquet`                             | Edge table (`edge_id`, `u`, `v`, `kind`, optional `geometry_wkt`, optional `length_m`). |
| `data/<dataset>/meta.json`                                 | Dataset metadata.                                                                       |
| `data/<dataset>/runs/<run>/node_metrics.parquet`           | Node features for analysis/ML.                                                          |
| `data/<dataset>/runs/<run>/edge_metrics.parquet`           | Edge features for analysis.                                                             |
| `data/<dataset>/runs/<run>/simulation_nodes.parquet`       | Node-level cascade outcomes.                                                            |
| `data/<dataset>/runs/<run>/simulation_steps.parquet`       | Step-by-step cascade trace.                                                             |
| `data/<dataset>/runs/<run>/{config,summary,manifest}.json` | Reproducible run metadata.                                                              |
| `data/<dataset>/predictions/risk_*.parquet`                | Predictions (`node_id`, `risk`).                                                        |

## Install (uv)

Core (CLI + graph + simulation):

`uv pip install -e .`

Enable ingestion (OSMnx + geo stack):

`uv pip install -e .[geo]`

Enable ML (Lightning + Torch Geometric):

`uv pip install -e .[ml]`

Dev tools (tests/lint/type-check):

`uv pip install -e .[dev]`

## CLI Usage

Ingest a city dataset:

`georesilience ingest city "Austin, TX, USA" --out data/austin`

Run a cascade simulation (prints bottleneck tables; writes a run folder):

`georesilience simulate cascade data/austin --alpha 0.2 --initial-failures 2 --mode targeted --seed 42`

Train a GCN (uses the latest run by default):

`georesilience train gcn data/austin --epochs 30 --seed 42`

Predict risk probabilities (prints summary + top-k; writes Parquet):

`georesilience predict gcn data/austin --checkpoint data/austin/models/gcn/last.ckpt`

Optional: select a specific simulation run for features:

`georesilience predict gcn data/austin --checkpoint data/austin/models/gcn/last.ckpt --run-dir data/austin/runs/<run_name>`

## Windows Note (Torch Geometric)

Torch Geometric wheels must match your installed PyTorch build (version + CPU/CUDA). If `uv pip install -e .[ml]` fails, install PyTorch first, then install Torch Geometric following the official wheel instructions for your exact torch/CUDA combination.
