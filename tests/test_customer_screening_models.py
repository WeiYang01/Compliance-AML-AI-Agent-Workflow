"""The customer portal is the only live screening path, so it must hand the
loaded ML models to the pipeline. Without this, Nodes 6a/6b are loaded nowhere
that matters and silently never run on real applications."""
from __future__ import annotations

import os

from fastapi.testclient import TestClient

from fraud_detect import customer_app
from fraud_detect.pipeline import run_pipeline as real_run_pipeline
from fraud_detect.store import CaseStore
from fraud_detect.synthetic import generate_clean


def test_apply_passes_loaded_models_to_the_pipeline(monkeypatch, tmp_path):
    txn_sentinel, stmt_sentinel = object(), object()
    monkeypatch.setattr(customer_app, "TXN_MODEL", txn_sentinel)
    monkeypatch.setattr(customer_app, "STATEMENT_MODEL", stmt_sentinel)
    monkeypatch.setattr(customer_app, "UPLOAD_DIR", tmp_path.resolve())
    monkeypatch.setattr(customer_app, "store", CaseStore(
        os.path.join(tmp_path, "cases.json"),
        audit_path=os.path.join(tmp_path, "audit_chain.jsonl"),
    ))

    seen = {}

    def spy(path, store=None, **kwargs):
        seen.update(kwargs)
        # Screen for real, but without the fake models.
        kwargs.update(txn_model=None, statement_model=None)
        return real_run_pipeline(path, store=store, **kwargs)

    monkeypatch.setattr(customer_app, "run_pipeline", spy)

    pdf = generate_clean(str(tmp_path / "samples")).path
    with open(pdf, "rb") as fh:
        r = TestClient(customer_app.app).post(
            "/apply",
            data={"company_name": "Acme", "uen": "201812345A",
                  "contact_email": "a@example.com"},
            files={"file": ("statement.pdf", fh, "application/pdf")},
            follow_redirects=False,
        )

    assert r.status_code == 303
    assert seen["statement_model"] is stmt_sentinel
    assert seen["txn_model"] is txn_sentinel
