"""GeoResilience package.

Primary entrypoint is the CLI (Typer) exposed via the `georesilience` console script.
"""

from __future__ import annotations

__all__ = ["__version__", "main"]

__version__ = "0.1.0"


def main() -> None:
    # Backwards-compatible hook if an older script still points at georesilience:main.
    from georesilience.cli.app import main as _main

    _main()
