from __future__ import annotations

import shutil
import time
from pathlib import Path


def inspect_staging(
    output_root: str | Path, older_than_hours: float = 24.0
) -> list[dict[str, object]]:
    """List interrupted unpublished runs without deleting them."""
    if older_than_hours < 0:
        raise ValueError("older_than_hours must be non-negative.")
    root = Path(output_root)
    if not root.exists():
        return []
    now = time.time()
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob(".*.partial")):
        if not path.is_dir():
            continue
        age_hours = (now - path.stat().st_mtime) / 3600.0
        rows.append(
            {
                "path": str(path),
                "age_hours": round(age_hours, 3),
                "eligible_for_cleanup": age_hours >= older_than_hours,
                "failure_manifest_present": (path / "failure_manifest.json").exists(),
            }
        )
    return rows


def cleanup_staging(
    output_root: str | Path,
    older_than_hours: float = 24.0,
    apply: bool = False,
) -> list[dict[str, object]]:
    """Inspect or explicitly remove stale unpublished run directories."""
    rows = inspect_staging(output_root, older_than_hours=older_than_hours)
    for row in rows:
        if apply and bool(row["eligible_for_cleanup"]):
            shutil.rmtree(str(row["path"]))
            row["action"] = "removed"
        else:
            row["action"] = "would_remove" if bool(row["eligible_for_cleanup"]) else "retained"
    return rows
