from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
from rich.console import Console
from rich.table import Table

from georesilience.io.parquet import write_parquet
from georesilience.ml.dataset import build_pyg_dataset
from georesilience.ml.deps import require_ml_deps
from georesilience.ml.lightning_module import GCNNodeRiskLightningModule
from georesilience.ml.train import _latest_run_dir


def predict_node_risk(
    *,
    dataset_dir: Path,
    run_dir: Path | None,
    checkpoint: Path,
    out_path: Path | None,
    top_k: int,
    device: str,
    console: Console,
) -> Path:
    require_ml_deps()
    import torch

    run_dir = run_dir or _latest_run_dir(dataset_dir)
    built = build_pyg_dataset(dataset_dir=dataset_dir, run_dir=run_dir, label_from_simulation=False)
    data: Any = built.data

    module = GCNNodeRiskLightningModule.load_from_checkpoint(str(checkpoint))
    module.eval()

    dev = "cuda" if device == "cuda" else "cpu"
    if device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    module.to(dev)
    x = data.x.to(dev)
    edge_index = data.edge_index.to(dev)

    with torch.no_grad():
        logits = module(x, edge_index)
        probs = torch.sigmoid(logits).detach().cpu().numpy()

    df = pl.DataFrame({"node_id": built.node_ids, "risk": probs})

    out = out_path or (dataset_dir / "predictions" / f"risk_{checkpoint.stem}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(df, out)

    # Rich table summary
    top = df.sort("risk", descending=True).head(top_k)
    table = Table(title=f"Top {top_k} risk nodes")
    table.add_column("node_id")
    table.add_column("risk", justify="right")
    for row in top.iter_rows(named=True):
        table.add_row(str(row["node_id"]), f"{float(row['risk']):.4f}")
    console.print(table)

    return out
