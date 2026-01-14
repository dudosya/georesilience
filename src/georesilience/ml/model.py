from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn as nn
    from torch_geometric.nn import GCNConv
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "ML features require optional dependencies. Install with: uv pip install -e .[ml]"
    ) from e


class GCNNodeClassifier(nn.Module):
    def __init__(self, *, in_channels: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        logits = self.lin(x).squeeze(-1)
        return logits


def build_gcn(*, in_channels: int, hidden_dim: int, dropout: float) -> Any:
    return GCNNodeClassifier(in_channels=in_channels, hidden_dim=hidden_dim, dropout=dropout)
