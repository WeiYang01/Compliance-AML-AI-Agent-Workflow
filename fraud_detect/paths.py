"""Filesystem locations the live app reads from.

Paths are anchored to the repository root, not the current working directory,
so the models load no matter which folder the server is started from. A
relative ``models/...`` path would silently resolve to nothing when uvicorn is
started elsewhere, and because the model loaders are fail-safe, the ML nodes
would then switch themselves off without any error.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
