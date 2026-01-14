from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console


@dataclass(frozen=True)
class AppContext:
    data_dir: Path
    console: Console


def get_context(ctx: object) -> AppContext:
    if not isinstance(ctx, dict):
        raise TypeError("Typer ctx.obj must be a dict")
    data_dir = ctx.get("data_dir")
    if not isinstance(data_dir, Path):
        raise TypeError("ctx.obj['data_dir'] must be a pathlib.Path")
    return AppContext(data_dir=data_dir, console=Console())
