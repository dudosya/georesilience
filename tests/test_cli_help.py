from __future__ import annotations

from typer.testing import CliRunner

from georesilience.cli.app import app


def test_help_includes_ml_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "train" in result.stdout
    assert "predict" in result.stdout
