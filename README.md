# GeoResilience

High-performance CLI tool that creates a “Digital Twin” of urban power grids to simulate infrastructure failure and predict risk.

## Architecture

```mermaid
flowchart LR
	CLI[Typer CLI + Rich UX]

	subgraph Data[Data Pipeline]
		OSM[OSMnx Fetch]
		Clean[Clean + Validate\n(Pydantic + Polars)]
		Store[(Parquet + JSON)]
	end

	subgraph Engine[Resilience Engine]
		Build[Build Graph\n(NetworkX)]
		Metrics[Centrality + Bottlenecks]
		Cascade[Cascading Failure Simulation]
	end

	subgraph ML[Predictive Risk Layer]
		Feat[Feature Engineering]
		GCN[GCN (PyTorch Geometric + Lightning)]
		Risk[Risk Scores]
	end

	CLI --> OSM --> Clean --> Store
	Store --> Build --> Metrics --> Cascade
	Store --> Feat --> GCN --> Risk
	Cascade --> Feat
	CLI --> Cascade
	CLI --> Risk
```

## Install (uv)

Core (CLI + graph + simulation):

`uv pip install -e .`

Enable ingestion from OpenStreetMap (OSMnx + geospatial stack):

`uv pip install -e .[geo]`

Enable ML (PyTorch Lightning + Torch Geometric):

`uv pip install -e .[ml]`

Dev tools (tests/lint/type-check):

`uv pip install -e .[dev]`

## Usage

Ingest a city dataset (writes `nodes.parquet`, `edges.parquet`, `meta.json`):

`georesilience ingest city "Austin, TX, USA" --out data/austin`

Run a cascading failure simulation:

`georesilience simulate cascade data/austin --alpha 0.2 --initial-failures 2 --mode targeted`

Outputs are written under `data/austin/runs/...`.

## Windows note (Torch Geometric)

Torch Geometric installs are sensitive to your PyTorch build (CPU vs CUDA) and version.
The most reliable path is:

1. Install PyTorch first (CPU/CUDA) using the official selector.
2. Install `torch-geometric` and its compiled companions matching that torch build.

If that becomes painful, we can swap the GCN implementation to pure PyTorch while keeping the same CLI/data contracts.
