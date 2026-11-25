"""Node 2 typography checks (fraud_detect/nodes/typography_forensics.py).

They must stay silent on genuine statements (including ones typeset unevenly throughout)
and on tampers that do not re-type a value, and name the values a careful in-place edit
re-typed.
"""
from __future__ import annotations

import random
import re

import pytest

from fraud_detect.corpus.schema import LAYOUTS, FileOp
from fraud_detect.corpus.seeding import rng
from fraud_detect.corpus.tamper.catalog import CATALOG
from fraud_detect.corpus.tamper.pdf_ops import apply_file_ops
from fraud_detect.ml import statement_geometry
from fraud_detect.ml.residual_eval import kern_everything
from fraud_detect.models import Routing, Severity
from fraud_detect.nodes import typography_forensics as tf
from fraud_detect.nodes.visual_forensics import AGENT as NODE2
from fraud_detect.pipeline import run_pipeline
from tests.corpus.helpers import DEFAULT_STYLE, load_renderer, make_test_ledger

TYPOGRAPHY = {"VF_SPACING_OUTLIER", "VF_COLUMN_MISALIGNED"}


def _genuine(layout="generator_v1", seed=5):
    return load_renderer(layout).render(make_test_ledger(3, seed), DEFAULT_STYLE, seed)


def _careful(variant, layout="generator_v1"):
    renderer = load_renderer(layout)
    for seed in range(40):
        plan = CATALOG["careful_edit"].plan(make_test_ledger(3, seed), renderer,
                                            rng(seed, "typo-test", "careful_edit"))
        if plan.label.notes.startswith(variant):
            base = renderer.render(plan.ledger, DEFAULT_STYLE, seed)
            return plan, apply_file_ops(base.pdf, plan.file_ops, base.row_map, seed)
    pytest.skip(f"no seed gives variant {variant}")


@pytest.mark.parametrize("layout", LAYOUTS)
def test_genuine_statements_raise_nothing(layout):
    for seed, months in ((3, 2), (8, 4)):
        pdf = load_renderer(layout).render(make_test_ledger(months, seed), DEFAULT_STYLE, seed).pdf
        assert tf.analyze_bytes(pdf) == []


def test_statement_typeset_unevenly_throughout_raises_nothing():
    assert tf.analyze_bytes(kern_everything(_genuine().pdf, random.Random(1))) == []


@pytest.mark.parametrize("name", ["genuine_resaved", "edited_amount", "font_mismatch",
                                  "structuring", "turnover_inflation", "cross_month_copy"])
def test_tampers_that_do_not_retype_a_value_raise_nothing(name):
    renderer = load_renderer("two_col_dmon")
    plan = CATALOG[name].plan(make_test_ledger(3, 2), renderer, rng(2, "typo-test", name))
    base = renderer.render(plan.ledger, DEFAULT_STYLE, 2)
    assert tf.analyze_bytes(apply_file_ops(base.pdf, plan.file_ops, base.row_map, 2)) == []


def test_retyped_value_is_reported_by_name():
    plan, pdf = _careful("retypeset")
    [f] = tf.analyze_bytes(pdf)
    assert f.code == "VF_SPACING_OUTLIER" and f.severity == Severity.MEDIUM and f.agent == NODE2
    assert re.search(r"\d,\d{3}\.\d{2}", f.evidence)
    assert any(op.new.split()[-1].replace("S$", "").replace("SGD", "") in f.evidence
               for op in plan.file_ops)


def test_value_off_its_column_is_reported_by_name():
    base = _genuine()
    row_id = next(r for r, cells in base.row_map.cells.items()
                  if re.search(r"\d,\d{3}\.\d{2}", cells["amount"].text))
    longer = re.sub(r"(\d),(\d{3}\.\d{2})", r"1\1,\2", base.row_map.cells[row_id]["amount"].text, count=1)
    op = FileOp(op="rewrite", row_id=row_id, field="amount", new=longer,
                params=(("variant", "anchored"),))
    [f] = tf.analyze_bytes(apply_file_ops(base.pdf, (op,), base.row_map, 5))
    assert f.code == "VF_COLUMN_MISALIGNED" and f.severity == Severity.MEDIUM
    assert re.search(r"\d\.\d{2}", f.evidence)


def test_rule_and_model_features_measure_the_same_thing():
    _, pdf = _careful("retypeset")
    features = statement_geometry.extract(pdf)
    codes = {f.code for f in tf.analyze_bytes(pdf)}
    assert (features["spacing_irregular_tokens"] > 0) == ("VF_SPACING_OUTLIER" in codes)
    assert (features["column_misaligned_tokens"] > 0) == ("VF_COLUMN_MISALIGNED" in codes)


def test_unreadable_file_raises_nothing_here():
    # structure_forensics already reports a file it cannot examine; no duplicate finding.
    assert tf.analyze_bytes(b"%PDF-1.4 not a pdf") == []


def test_pipeline_routes_a_retyped_statement_to_a_person(tmp_path):
    _, pdf = _careful("retypeset")
    path = tmp_path / "careful.pdf"
    path.write_bytes(pdf)
    report = run_pipeline(str(path))
    assert TYPOGRAPHY & {f.code for f in report.findings}
    assert report.routing == Routing.HUMAN_REVIEW
