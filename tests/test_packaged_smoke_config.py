from __future__ import annotations

from pathlib import Path

import pytest

from evalforge.config import load_config, write_packaged_smoke_config


def test_packaged_smoke_config_matches_checked_in_smoke_config(tmp_path: Path) -> None:
    exported = write_packaged_smoke_config(tmp_path / "smoke.yaml")
    root = Path(__file__).resolve().parents[1]
    assert load_config(exported) == load_config(root / "configs" / "smoke.yaml")


def test_packaged_smoke_config_refuses_implicit_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "smoke.yaml"
    write_packaged_smoke_config(destination)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_packaged_smoke_config(destination)
    assert write_packaged_smoke_config(destination, overwrite=True) == destination
