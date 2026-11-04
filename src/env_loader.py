"""Lightweight zero-dependency .env loader for local configuration."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(env_path: str | Path | None = None) -> None:
    """Load key=value pairs from `.env` into `os.environ` without overwriting existing vars."""
    path = Path(env_path) if env_path is not None else Path(__file__).resolve().parent.parent / ".env"
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and (key not in os.environ or not os.environ[key]):
            os.environ[key] = val
