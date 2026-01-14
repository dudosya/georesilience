from __future__ import annotations


def require_ml_deps() -> None:
    try:
        import pytorch_lightning  # noqa: F401
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "ML features require optional dependencies. Install with: uv pip install -e .[ml]"
        ) from e
