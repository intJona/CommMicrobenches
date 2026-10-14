"""Disk-backed cache for the CommBench results scan.

Lives in .cache/ next to dashboard.py so only the first-ever scan pays the
network-share cost -- a restart or explicit Refresh click re-reads the JSON
instantly instead of re-globbing RESULTS_ROOT.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

CACHE_DIR = Path(__file__).parent / ".cache"


def _cache_path(results_root: Path) -> Path:
    key = hashlib.sha1(str(results_root).encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"runs_{key}.json"


def load_runs(results_root: Path) -> tuple[list[dict] | None, dict | None]:
    """Load cached runs and metadata. Returns (runs, meta) or (None, None)."""
    path = _cache_path(results_root)
    if not path.is_file():
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None, None
    return payload.get("runs"), payload.get("meta")


def save_runs(results_root: Path, runs: list[dict], meta: dict) -> None:
    """Best-effort write: failures never crash the caller."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(results_root)
    tmp_path = path.with_suffix(f".tmp{time.time_ns()}")
    payload = {"runs": runs, "meta": meta}
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        tmp_path.replace(path)
    except OSError:
        tmp_path.unlink(missing_ok=True)


def make_meta(started_at: float) -> dict:
    """Create metadata dict with generation timestamp and duration."""
    now = time.time()
    return {"generated_at": now, "duration_seconds": round(now - started_at, 1)}
