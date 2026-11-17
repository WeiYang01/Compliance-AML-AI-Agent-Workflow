"""The live app must find its models whichever folder the server starts in.

The loaders are fail-safe (a missing file returns ``None`` and the ML node is
skipped), so a path that resolves against the working directory would turn the
ML nodes off silently when the server is started outside the repo root.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pytest

from fraud_detect.ml.gnn import gnn_lookup
from fraud_detect.nodes import ml_statement, ml_transaction
from fraud_detect.paths import MODELS_DIR

REPO = Path(__file__).resolve().parent.parent


def test_models_dir_is_the_repo_models_folder():
    assert MODELS_DIR == REPO / "models"
    assert MODELS_DIR.is_absolute()


@pytest.mark.parametrize("load, filename", [
    (ml_statement.load_model, "statement_forgery_clf.joblib"),
    (ml_transaction.load_model, "txn_fraud_clf.joblib"),
    (lambda: ml_transaction.load_live_model(env={ml_transaction.ENABLE_ENV: "1"}),
     "txn_fraud_clf.joblib"),
    (gnn_lookup.load_scores, "gnn_scores.joblib"),
])
def test_default_paths_ignore_the_working_directory(monkeypatch, tmp_path, load, filename):
    monkeypatch.chdir(tmp_path)  # start "the server" somewhere else
    asked_for = []

    def fake_load(path, *args, **kwargs):
        asked_for.append(Path(path))
        raise FileNotFoundError(path)  # keep the fail-safe path; no real model needed

    monkeypatch.setattr(joblib, "load", fake_load)
    assert load() is None
    assert asked_for == [REPO / "models" / filename]
