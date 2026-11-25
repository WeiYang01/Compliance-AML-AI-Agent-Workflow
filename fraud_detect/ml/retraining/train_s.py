"""Node 6a candidate on the statement feature set (one score per statement).

Mirrors ``train_a``: fit on the train split only, choose cutoffs on validation statement
scores only, refuse any input that is not an audited statement feature. The bundle records
``feature_set = "statement"`` so the live node and the evaluation score it per statement.
"""
from __future__ import annotations

import io
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib

from fraud_detect.ml.retraining import safe_io
from fraud_detect.ml.retraining.audit import AUDIT_V2, STATEMENT_NODE, covers_statement, model_input_columns
from fraud_detect.ml.retraining.cutoffs import CutoffRule, choose_cutoffs, operating_curve
from fraud_detect.ml.retraining.features_a import ForbiddenFeatureError
from fraud_detect.ml.statement_behaviour import STAGE_B_FEATURES

MODEL_LABEL = "model trained on synthetic data"
FEATURE_SET = "statement"
LEVEL = "document"           # one score per statement: the statement IS the document
LIVE_CUTOFF_NOTE = ("A statement-level Node 6a model: the live node scores the whole statement "
                    "once and uses the stored cutoffs directly.")


@dataclass
class TrainedS:
    path: Path
    bundle: dict


def make_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.1,
                                          random_state=0)


def train_statement_candidate(table, *, out_path, audit=AUDIT_V2, rule=None, master_seed=None,
                              held_out_layout=None, left_out_no_fraud_row=0,
                              is_ignored=None) -> TrainedS:
    """``table`` is ``features_s.StatementTable.frame`` (train and validation rows)."""
    if not covers_statement(audit):
        raise ValueError(f"audit {audit.version} does not review the statement feature set")
    rule = rule if rule is not None else CutoffRule(level=LEVEL)
    if rule.level != LEVEL:
        raise ValueError(f"the statement model uses level {LEVEL!r}, got {rule.level!r}")
    columns = model_input_columns(audit, STATEMENT_NODE)
    offenders = [c for c in columns if c not in STAGE_B_FEATURES]
    if offenders:
        raise ForbiddenFeatureError(offenders)
    out_path = Path(out_path)
    safe_io.ensure_ignored(out_path, is_ignored=is_ignored)

    train = table[table["split"] == "train"]
    valid = table[table["split"] == "validation"]
    model = make_model().fit(train[columns].to_numpy(), train["y"].astype(int).to_numpy())

    scores = model.predict_proba(valid[columns].to_numpy())[:, 1]
    genuine = (valid["y"] == 0).tolist()
    cuts = choose_cutoffs(list(map(float, scores)), genuine, rule)
    thresholds = {"medium": cuts.medium, "high": cuts.high}
    curve = [asdict(r) for r in operating_curve(list(map(float, scores)), genuine, thresholds)]

    bundle = {
        "pipeline": model,
        "feature_set": FEATURE_SET,
        "feature_columns": columns,
        "thresholds": thresholds,
        "cutoff_rule": {"target_fa_medium": rule.target_fa_medium,
                        "target_fa_high": rule.target_fa_high, "level": rule.level},
        "live_cutoff_note": LIVE_CUTOFF_NOTE,
        "left_out_no_fraud_row": int(left_out_no_fraud_row),
        "validation_curve": curve,
        "audit_version": audit.version,
        "audit_digest": audit.digest(),
        "master_seed": master_seed,
        "held_out_layout": held_out_layout,
        "model_label": MODEL_LABEL,
        "synthetic_data": True,
    }
    buf = io.BytesIO()
    joblib.dump(bundle, buf)
    safe_io.atomic_write(out_path, buf.getvalue(), is_ignored=is_ignored)
    return TrainedS(out_path, bundle)


def score_statements(bundle: dict, frame) -> list:
    """Probability per statement row of ``frame`` (columns from the bundle)."""
    return [float(p) for p in bundle["pipeline"].predict_proba(
        frame[list(bundle["feature_columns"])].to_numpy())[:, 1]]
