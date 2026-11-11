"""
Automatic Two-Way Telemetry & Console Log Sync between Home PC and Laptop via GitHub.

Why this exists:
  - Antigravity is locked to the user's organization laptop (`nikmuhammadshahiran`),
    while MetaTrader 5 and the RTX 5060 GPU run on the user's Home PC (`PRO B760M-A WIFI`).
  - This module writes sanitized diagnostics (`logs/home_pc_status.json`) and recent
    console output (`logs/home_pc_console.txt`) and pushes them to GitHub automatically,
    eliminating the need to take screenshots and send them over WhatsApp.
  - Strictly excludes any secrets, tokens, or account credentials.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"
STATUS_JSON_PATH = LOGS_DIR / "home_pc_status.json"
CONSOLE_TXT_PATH = LOGS_DIR / "home_pc_console.txt"


def _scrub_secrets(text: str) -> str:
    """Ensure bot tokens or sensitive strings are never written to sync logs."""
    return re.sub(r"\d{8,12}:[A-Za-z0-9_-]{30,}", "[REDACTED_TOKEN]", text)


def collect_gpu_diagnostics() -> dict[str, Any]:
    """Inspect PyTorch, CUDA version, and RTX 5060 (sm_120) kernel execution status."""
    info: dict[str, Any] = {
        "torch_installed": False,
        "torch_version": None,
        "cuda_version": None,
        "cuda_available": False,
        "gpu_name": None,
        "cuda_kernel_ok": False,
        "cuda_error": None,
    }
    try:
        import torch  # type: ignore

        info["torch_installed"] = True
        info["torch_version"] = getattr(torch, "__version__", "unknown")
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            try:
                _t = torch.zeros(1, device="cuda:0") + 1.0
                info["cuda_kernel_ok"] = bool(float(_t.item()) == 1.0)
            except Exception as exc:
                info["cuda_kernel_ok"] = False
                info["cuda_error"] = str(exc)
    except Exception as exc:
        info["cuda_error"] = str(exc)
    return info


def write_and_push_telemetry(
    console_lines: list[str],
    extra_status: dict[str, Any] | None = None,
    push_to_github: bool = True,
) -> bool:
    """
    Synchronize with `origin/main` (conflict-free), write `logs/home_pc_status.json` and
    `logs/home_pc_console.txt`, then commit and push to GitHub.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    code_updated = False
    if push_to_github:
        try:
            subprocess.run(
                ["git", "rebase", "--abort"],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
            )
            subprocess.run(
                ["git", "fetch", "origin", "main"],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                timeout=25,
            )
            diff_res = subprocess.run(
                ["git", "diff", "--name-only", "HEAD..origin/main"],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            changed_files = [
                f.strip()
                for f in (diff_res.stdout or "").splitlines()
                if f.strip() and not f.strip().startswith("logs/")
            ]
            if changed_files:
                code_updated = True
            subprocess.run(
                ["git", "pull", "--rebase", "--autostash", "origin", "main"],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                timeout=15,
            )
        except Exception:
            pass

    status_payload: dict[str, Any] = {
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "hostname": platform.node(),
        "python_version": sys.version.split()[0],
        "gpu": collect_gpu_diagnostics(),
        "pending_hot_reload": code_updated,
    }
    if extra_status:
        status_payload.update(extra_status)

    STATUS_JSON_PATH.write_text(
        json.dumps(status_payload, indent=2),
        encoding="utf-8",
    )

    if extra_status and isinstance(extra_status.get("learned_weights"), dict):
        lw = extra_status["learned_weights"]
        (LOGS_DIR / "learned_weights.json").write_text(
            json.dumps(lw, indent=2), encoding="utf-8"
        )
        inbox_payload = {
            "updated_utc": status_payload["updated_utc"],
            "hostname": status_payload["hostname"],
            "live_equity": extra_status.get("live_equity"),
            "bot_stats": extra_status.get("bot_stats"),
            "symbol_tuning": lw.get("symbol_tuning", {}),
            "symbol_streaks": lw.get("symbol_streaks", {}),
            "pending_post_mortems": lw.get("pending_post_mortems", []),
            "ai_feedback_feed": (lw.get("ai_feedback_feed") or [])[-25:],
        }
        (LOGS_DIR / "ai_feedback_inbox.json").write_text(
            json.dumps(inbox_payload, indent=2), encoding="utf-8"
        )

    sanitized_lines = [_scrub_secrets(line) for line in console_lines[-150:]]
    CONSOLE_TXT_PATH.write_text(
        "\n".join(sanitized_lines) + "\n",
        encoding="utf-8",
    )

    if not push_to_github:
        return True

    try:
        subprocess.run(
            ["git", "config", "user.name", "DataEngine-HomePC"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "homepc@data-engine.local"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
        )
        subprocess.run(
            [
                "git",
                "add",
                "-f",
                "logs/home_pc_status.json",
                "logs/home_pc_console.txt",
                "logs/ai_feedback_inbox.json",
                "logs/learned_weights.json",
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
        )
        commit_res = subprocess.run(
            ["git", "commit", "-m", f"Sync Home PC telemetry ({platform.node()})"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if commit_res.returncode == 0:
            push_res = subprocess.run(
                ["git", "push", "origin", "main"],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            ok = push_res.returncode == 0
        else:
            ok = True
        if code_updated and "--once" not in sys.argv:
            print("  [Auto-Update] New code detected from GitHub! Hot-reloading bot automatically...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        return ok
    except Exception:
        return False
    return False
