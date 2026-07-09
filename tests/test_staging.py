from __future__ import annotations

import os
import time
from pathlib import Path

from evalforge.staging import cleanup_staging, inspect_staging


def test_staging_cleanup_is_dry_run_by_default(tmp_path: Path) -> None:
    stale = tmp_path / ".interrupted.partial"
    stale.mkdir()
    old = time.time() - 7200
    os.utime(stale, (old, old))
    inspected = inspect_staging(tmp_path, older_than_hours=1)
    assert inspected[0]["eligible_for_cleanup"] is True
    dry_run = cleanup_staging(tmp_path, older_than_hours=1, apply=False)
    assert dry_run[0]["action"] == "would_remove"
    assert stale.exists()
    cleaned = cleanup_staging(tmp_path, older_than_hours=1, apply=True)
    assert cleaned[0]["action"] == "removed"
    assert not stale.exists()
