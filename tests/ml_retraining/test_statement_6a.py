"""Node 6a statement feature set in Stage B (features_s, train_s, audit-v2, runner, evaluation).

The default feature set stays "pdf": a run configured without the option keeps its run id
and its audit-v1. Choosing "statement" trains one score per statement on the audited
statement features, runs its own Leak_Test, and is scored per statement in compare.
"""
from __future__ import annotations

import datetime as dt
import json

import joblib
import pandas as pd
import pytest

from fraud_detect.ml.retraining.audit import (
    AUDIT_V1, AUDIT_V2, STATEMENT_NODE, AuditError, FeatureAudit, covers_statement, model_input_columns,
    validate)
from fraud_detect.ml.retraining.features_s import build_statement_table, records_from_rows
from fraud_detect.ml.retraining.runner import CliRunConfig
from fraud_detect.ml.retraining.train_s import train_statement_candidate
from fraud_detect.ml.statement_behaviour import NOT_COMPUTABLE, STAGE_B_FEATURES, stage_b_features

from .test_cli import _copy_corpora, _fake_models, Env, prepared, shared_cache  # noqa: F401
from .conftest import mini_corpus  # noqa: F401


# ---- the feature set ---------------------------------------------------------------------

def test_stage_b_features_need_only_dates_and_amounts():
    recs = [(dt.date(2026, m, d), a) for m in (1, 2, 3)
            for d, a in ((3, 1000.5), (9, -300.2), (15, 2200.1), (20, -800.0))]
    f = stage_b_features(recs)
    assert list(f) == STAGE_B_FEATURES
    assert f["benford_deviation"] == NOT_COMPUTABLE          # fewer than 20 receipts
    assert stage_b_features(recs[:3]) is None                # too few transactions


def test_round_amounts_and_payer_features_are_left_out():
    assert "round_amount_share" not in STAGE_B_FEATURES      # a generator tell in the audit
    assert not {"top_payer_share", "one_month_payer_share",
                "repeat_across_months_share"} & set(STAGE_B_FEATURES)


# ---- the audit ----------------------------------------------------------------------------

def test_audit_v1_is_unchanged_and_v2_extends_it():
    validate(AUDIT_V1)
    validate(AUDIT_V2)
    assert not covers_statement(AUDIT_V1) and covers_statement(AUDIT_V2)
    assert AUDIT_V2.entries[:len(AUDIT_V1.entries)] == AUDIT_V1.entries
    assert model_input_columns(AUDIT_V2, STATEMENT_NODE) == STAGE_B_FEATURES
    assert model_input_columns(AUDIT_V2, "6a") == model_input_columns(AUDIT_V1, "6a")


def test_partial_statement_audit_is_rejected():
    partial = FeatureAudit("audit-partial", AUDIT_V2.entries[:-1])
    with pytest.raises(AuditError, match="credits_per_month_peak_ratio"):
        validate(partial)


# ---- the config ----------------------------------------------------------------------------

def test_default_feature_set_keeps_the_run_id():
    default = CliRunConfig()
    assert "node6a_feature_set" not in default.canonical_json()
    statement = CliRunConfig(node6a_feature_set="statement")
    assert json.loads(statement.canonical_json())["node6a_feature_set"] == "statement"
    assert statement.run_id != default.run_id


# ---- table and training ------------------------------------------------------------------------

class _Doc:
    def __init__(self, doc_id, split, tampered, layout="generator_v1"):
        self.corpus_key, self.doc_id, self.split = "c1", doc_id, split
        self.is_tampered, self.layout = tampered, layout


def _rows(doc_id, inflate=0.0, fraud=False, seed=0):
    import random
    rnd = random.Random(seed)
    out = []
    for m in (1, 2, 3):
        for d, a in ((2, 1500.0), (6, -400.0), (11, 2300.0), (17, -900.0), (23, 700.0), (27, -350.0)):
            a = round(a * rnd.uniform(0.6, 1.4), 2) if seed else a   # genuine month-to-month variation
            amount = a * (1 + inflate) if a > 0 and m == 2 else a
            out.append({"corpus_key": "c1", "doc_id": doc_id, "timestamp": f"2026-0{m}-{d:02d}",
                        "direction": "CR" if amount >= 0 else "DR", "amount_sgd": abs(amount),
                        "y": int(fraud and m == 2 and a > 0)})
    return out


def test_statement_table_labels_and_leaves_out_untouched_tampered_statements():
    rows = pd.DataFrame(_rows("g") + _rows("t", inflate=1.5, fraud=True) + _rows("d"))
    docs = [_Doc("g", "train", False), _Doc("t", "train", True), _Doc("d", "train", True)]
    table = build_statement_table(rows, docs)
    assert table.left_out_no_fraud_row == 1                  # "d": tampered, transactions untouched
    assert dict(zip(table.frame["doc_id"], table.frame["y"])) == {"g": 0, "t": 1}
    assert list(table.frame.columns[4:-1]) == STAGE_B_FEATURES


def test_records_from_rows_restores_the_sign():
    rec = records_from_rows(pd.DataFrame(_rows("g")[:2]))
    assert rec[0][1] > 0 and rec[1][1] < 0


def test_training_writes_a_statement_bundle_with_validation_cutoffs(tmp_path):
    rows, docs = [], []
    for k in range(120):
        split = "train" if k < 60 else "validation"
        tampered = k % 2 == 1
        rows += _rows(f"d{k}", inflate=1.5 + 0.1 * (k % 5) if tampered else 0.0, fraud=tampered,
                      seed=k + 1)
        docs.append(_Doc(f"d{k}", split, tampered))
    table = build_statement_table(pd.DataFrame(rows), docs)
    from fraud_detect.ml.retraining.cutoffs import CutoffRule
    rule = CutoffRule(target_fa_medium=20.0, target_fa_high=10.0, level="document")  # small toy data
    res = train_statement_candidate(table.frame, out_path=tmp_path / "6a.joblib", rule=rule,
                                    is_ignored=lambda p: True)
    bundle = joblib.load(tmp_path / "6a.joblib")
    assert bundle["feature_set"] == "statement" and bundle["audit_version"] == "audit-v2"
    assert bundle["feature_columns"] == STAGE_B_FEATURES
    assert 0.0 <= bundle["thresholds"]["medium"] <= bundle["thresholds"]["high"] <= 1.0
    assert res.bundle["cutoff_rule"]["level"] == "document"


def test_training_refuses_an_audit_without_the_statement_set(tmp_path):
    with pytest.raises(ValueError, match="does not review the statement feature set"):
        train_statement_candidate(pd.DataFrame(), out_path=tmp_path / "x.joblib", audit=AUDIT_V1,
                                  is_ignored=lambda p: True)


# ---- the live node -------------------------------------------------------------------------------

def test_live_node_scores_a_statement_bundle_once(tmp_path):
    from types import SimpleNamespace
    from fraud_detect.nodes import ml_transaction

    import numpy as np

    class Always:                       # a stand-in with sklearn's output type
        def predict_proba(self, X):
            return np.array([[0.05, 0.95] for _ in X])

    bundle = {"feature_set": "statement", "pipeline": Always(), "feature_columns": STAGE_B_FEATURES,
              "thresholds": {"medium": 0.5, "high": 0.9}, "model_label": "model trained on synthetic data"}
    txns = [SimpleNamespace(date=dt.date(2026, m, d), amount=a, description="X", page=1, index=i,
                            balance=0.0)
            for i, (m, d, a) in enumerate((m, d, a) for m in (1, 2, 3)
                                          for d, a in ((3, 1000.5), (9, -300.2), (15, 2200.1)))]
    ing = SimpleNamespace(transactions=txns, uen="201812345A")
    [f] = ml_transaction.analyze(ing, bundle).findings
    assert f.code == "ML_TXN_STATEMENT_RISK" and f.severity.value == "HIGH"


# ---- end to end: a plumbing run on the Mini corpora ------------------------------------------------

def test_plumbing_run_with_the_statement_feature_set(prepared, shared_cache, tmp_path):  # noqa: F811
    env = Env(tmp_path, shared_cache)
    _fake_models(env.models)
    _copy_corpora(prepared, env)
    assert env.init("--node6a-feature-set", "statement").code == 0
    run = json.loads((env.run_dir / "run.json").read_text())
    assert "statement" in json.dumps(run)
    assert env.invoke("check").code in (0, 3)
    leak = env.invoke("leak-test")
    assert "Node 6a statement features:" in leak.out, leak.out
    env.decide("waive-gate", "capability", reason="plumbing")
    env.decide("waive-gate", "subtle", reason="plumbing")
    train = env.invoke("train", "--node", "6a")
    assert train.code == 0, train.out
    bundle = joblib.load(env.run_dir / "release" / "6a.joblib")
    assert bundle["feature_set"] == "statement" and bundle["audit_version"] == "audit-v2"
    compare = env.invoke("compare")
    assert compare.code == 0, compare.out
    assert "Node 6a" in compare.out
    report = (env.run_dir / "comparison_report.md").read_text()
    for name in STAGE_B_FEATURES:  # the statement inputs appear in the report's audit table
        assert f"| {name} |" in report, name


# ---- the offline like-for-like comparison ----------------------------------------------------------

def test_like_for_like_cutoff_keeps_false_alarms_within_the_target():
    from fraud_detect.ml.retraining.like_for_like_6a import cutoff_at, metrics
    genuine = [i / 100 for i in range(100)]
    scores, labels = genuine + [0.995, 0.5], [0] * 100 + [1, 1]
    for fa in (0.05, 0.01):
        c = cutoff_at(scores, labels, fa)
        m = metrics(scores, labels, {"x": c})
        assert m["fa@x"] <= fa and m["fa@x"] >= fa - 0.01   # as many alerts as the target allows
    assert metrics(scores, labels, {"x": cutoff_at(scores, labels, 0.01)})["catch@x"] == 0.5


def test_like_for_like_summary_reports_both_models_per_set():
    from fraud_detect.ml.retraining.like_for_like_6a import METRICS, summarise
    ev = {f"{n}/{p}/{m}": dict({k: 0.5 for k in METRICS}, n_pos=1, n_neg=1)
          for n in ("test", "heldout_seed") for p in ("core", "all") for m in ("row", "statement")}
    doc = {"doc_id": "a", "y": 1, "pop": "core", "tamper": ["edited_amount"], "row": 0.9, "statement": 0.1}
    seed = {"cutoffs": {"row": {"fa5": 0.5}, "statement": {"fa5": 0.5}}, "eval": ev,
            "docs": {"heldout_seed": [doc, dict(doc, y=0, tamper=[])]}}
    text = summarise({1: seed, 2: seed})
    assert "synthetic data" in text and "heldout_seed / all" in text
    assert "edited_amount" in text and "row 1.00  statement 0.00  either 1.00" in text
