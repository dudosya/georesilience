from __future__ import annotations

from pathlib import Path
from typing import Any

from georesilience.ml.dataset import build_pyg_dataset
from georesilience.ml.deps import require_ml_deps
from georesilience.ml.lightning_module import GCNConfig, build_module


def _latest_run_dir(dataset_dir: Path) -> Path:
    runs = dataset_dir / "runs"
    if not runs.exists():
        raise FileNotFoundError(f"No runs directory found at {runs}")
    candidates = [p for p in runs.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {runs}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def train_node_risk_gcn(
    *,
    dataset_dir: Path,
    run_dir: Path | None,
    out_dir: Path | None,
    epochs: int,
    lr: float,
    hidden_dim: int,
    dropout: float,
    weight_decay: float,
    val_fraction: float,
    seed: int,
    device: str,
) -> Path:
    require_ml_deps()
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import ModelCheckpoint
    from torch_geometric.loader import DataLoader

    pl.seed_everything(seed, workers=True)

    run_dir = run_dir or _latest_run_dir(dataset_dir)

    built = build_pyg_dataset(dataset_dir=dataset_dir, run_dir=run_dir, label_from_simulation=True)
    data: Any = built.data

    # Create train/val masks
    n = int(data.num_nodes)
    idx = torch.randperm(n)
    split = int((1.0 - val_fraction) * n)
    train_idx = idx[:split]
    val_idx = idx[split:]

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    data.train_mask = train_mask
    data.val_mask = val_mask

    cfg = GCNConfig(
        feature_names=built.feature_names,
        hidden_dim=hidden_dim,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
    )

    module = build_module(config=cfg, x_mean=data.x_mean, x_std=data.x_std)

    ckpt_dir = out_dir or (dataset_dir / "models" / "gcn")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="gcn-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    accelerator = "auto" if device == "auto" else device
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=1,
        deterministic=True,
        callbacks=[checkpoint],
        enable_progress_bar=True,
        log_every_n_steps=1,
    )

    train_loader = DataLoader([data], batch_size=1)
    val_loader = DataLoader([data], batch_size=1)
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return ckpt_dir
