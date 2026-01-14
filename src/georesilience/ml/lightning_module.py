from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import pytorch_lightning as pl
    import torch
    import torch.nn.functional as F
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "ML features require optional dependencies. Install with: uv pip install -e .[ml]"
    ) from e

from georesilience.ml.model import build_gcn


@dataclass(frozen=True)
class GCNConfig:
    feature_names: list[str]
    hidden_dim: int
    dropout: float
    lr: float
    weight_decay: float


def _accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).to(y.dtype)
    return float((preds == y).float().mean().item())


class GCNNodeRiskLightningModule(pl.LightningModule):
    def __init__(
        self,
        *,
        feature_names: list[str],
        hidden_dim: int,
        dropout: float,
        lr: float,
        weight_decay: float,
        x_mean: torch.Tensor | None = None,
        x_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            {
                "feature_names": feature_names,
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "lr": lr,
                "weight_decay": weight_decay,
            }
        )
        self.model = build_gcn(
            in_channels=len(feature_names), hidden_dim=hidden_dim, dropout=dropout
        )

        # These buffers are persisted in checkpoints. They are optional at init-time
        # so `load_from_checkpoint()` can instantiate the module from saved hparams.
        n_features = len(feature_names)
        if x_mean is None:
            x_mean = torch.zeros(n_features, dtype=torch.float32)
        if x_std is None:
            x_std = torch.ones(n_features, dtype=torch.float32)

        x_mean = x_mean.detach().float().reshape(-1)
        x_std = x_std.detach().float().reshape(-1)
        if x_mean.numel() != n_features or x_std.numel() != n_features:
            raise ValueError(
                "x_mean/x_std must be 1D tensors with length equal to number of features "
                f"({n_features}); got mean={tuple(x_mean.shape)} std={tuple(x_std.shape)}"
            )

        self.register_buffer("x_mean", x_mean)
        self.register_buffer("x_std", x_std)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = (x - self.x_mean) / torch.where(self.x_std > 0, self.x_std, torch.ones_like(self.x_std))
        return self.model(x, edge_index)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        logits = self(batch.x, batch.edge_index)
        y = batch.y
        mask = batch.train_mask
        loss = F.binary_cross_entropy_with_logits(logits[mask], y[mask])
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", _accuracy_from_logits(logits[mask], y[mask]), prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        logits = self(batch.x, batch.edge_index)
        y = batch.y
        mask = batch.val_mask
        loss = F.binary_cross_entropy_with_logits(logits[mask], y[mask])
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", _accuracy_from_logits(logits[mask], y[mask]), prog_bar=True)
        return loss

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(
            self.parameters(),
            lr=float(self.hparams["lr"]),
            weight_decay=float(self.hparams["weight_decay"]),
        )


def build_module(
    *, config: GCNConfig, x_mean: torch.Tensor, x_std: torch.Tensor
) -> GCNNodeRiskLightningModule:
    return GCNNodeRiskLightningModule(
        feature_names=config.feature_names,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        lr=config.lr,
        weight_decay=config.weight_decay,
        x_mean=x_mean,
        x_std=x_std,
    )
