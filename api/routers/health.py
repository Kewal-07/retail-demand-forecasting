"""GET /health. See CLAUDE.md Section 10."""
import subprocess
from pathlib import Path

from fastapi import APIRouter, Request

router = APIRouter()


def _git_sha() -> str:
    try:
        repo_root = Path(__file__).resolve().parent.parent.parent
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


@router.get("/health")
def health(request: Request) -> dict:
    return {
        "model_version": ",".join(sorted(request.app.state.models.keys())),
        "git_sha": _git_sha(),
        "load_time": request.app.state.start_time,
    }
