"""Node 6a statement feature table: one row per statement (the "statement" feature set).

Built from the same rows.csv table as the Node 6a row model (``features_a``), grouped by
document. Each statement's dates and signed amounts go through
``statement_behaviour.stage_b_features``, the same function the live node uses on parsed
transactions, so training and screening compute identical values.

Labels follow the row model's statement sets (``train_a.detection_statements``):
genuine statements are 0, tampered statements with at least one fraud row are 1, and
tampered statements with no fraud row (document-only tampers such as careful_edit, which
leave the transactions untouched) are left out and counted, because nothing in their
transactions differs from a genuine statement. Only the audited, PDF-visible inputs
(dates and amounts) are read; ``kind`` and the fraud labels in rows.csv never reach a
feature.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fraud_detect.ml.statement_behaviour import STAGE_B_FEATURES, stage_b_features

KEY_COLS = ["corpus_key", "doc_id", "layout", "split"]


@dataclass
class StatementTable:
    frame: pd.DataFrame          # KEY_COLS + STAGE_B_FEATURES + ["y"]
    left_out_no_fraud_row: int   # tampered statements whose transactions are untouched
    too_short: int               # statements with too few transactions for the features


def records_from_rows(rows: pd.DataFrame) -> list:
    """(date, signed amount) pairs from row-table rows (timestamp, direction, amount_sgd)."""
    out = []
    for r in rows.itertuples(index=False):
        date = pd.Timestamp(r.timestamp).date() if r.timestamp else None
        amount = float(r.amount_sgd) if r.direction == "CR" else -float(r.amount_sgd)
        out.append((date, amount))
    return out


def build_statement_table(rows: pd.DataFrame, docs) -> StatementTable:
    """One feature row per statement.

    ``rows`` is ``RowTableA.rows`` (``features_a.rows_to_frame``); ``docs`` are the
    corpus documents (``corpus_key``, ``doc_id``, ``layout``, ``split``, ``is_tampered``).
    """
    fraud = {k for k, v in rows.groupby(["corpus_key", "doc_id"])["y"].max().items() if v == 1}
    grouped = {k: g for k, g in rows.groupby(["corpus_key", "doc_id"], sort=False)}
    records, left_out, short = [], 0, 0
    for d in docs:
        key = (d.corpus_key, d.doc_id)
        if key not in grouped:
            continue
        if d.is_tampered and key not in fraud:
            left_out += 1
            continue
        features = stage_b_features(records_from_rows(grouped[key]))
        if features is None:
            short += 1
            continue
        records.append({"corpus_key": d.corpus_key, "doc_id": d.doc_id, "layout": d.layout,
                        "split": d.split, **features, "y": int(bool(d.is_tampered))})
    frame = pd.DataFrame(records, columns=KEY_COLS + list(STAGE_B_FEATURES) + ["y"])
    return StatementTable(frame, left_out, short)
