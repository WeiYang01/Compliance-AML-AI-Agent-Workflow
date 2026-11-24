"""Step 3: the careful_edit tamper and the new Node 6a / 6b feature sets.

careful_edit must change values while leaving nothing the rule-based checks can see; the
typography features must notice an edited number without penalising a document that is
typeset unevenly throughout; the behaviour features must move with statement-level
fabrication.
"""
from __future__ import annotations

import io
import random
import re

import pdfplumber
import pytest

from fraud_detect.corpus.schema import EXTRA_TAMPER_TYPES, TAMPER_TYPES
from fraud_detect.corpus.seeding import rng
from fraud_detect.corpus.tamper.catalog import CATALOG
from fraud_detect.corpus.tamper.pdf_ops import apply_file_ops
from fraud_detect.ml import residual_eval, statement_behaviour, statement_geometry
from fraud_detect.nodes import financial_logic, structure_forensics, visual_forensics
from fraud_detect.nodes.ingestion import ingest
from tests.corpus.helpers import DEFAULT_STYLE, load_renderer, make_test_ledger


def _careful(layout="generator_v1", *, variant=None, seed_from=0):
    renderer = load_renderer(layout)
    for seed in range(seed_from, seed_from + 40):
        plan = CATALOG["careful_edit"].plan(make_test_ledger(3, seed), renderer,
                                            rng(seed, "step3-test", "careful_edit"))
        if variant and not plan.label.notes.startswith(variant):
            continue
        base = renderer.render(plan.ledger, DEFAULT_STYLE, seed)
        return plan, base, apply_file_ops(base.pdf, plan.file_ops, base.row_map, seed)
    pytest.skip(f"no seed gives careful_edit variant {variant}")


def _write(tmp_path, pdf, name="doc.pdf"):
    path = tmp_path / name
    path.write_bytes(pdf)
    return path


# ---- careful_edit -------------------------------------------------------------------

def test_careful_edit_is_an_opt_in_extra_so_default_corpora_are_unchanged():
    assert "careful_edit" in EXTRA_TAMPER_TYPES and "careful_edit" not in TAMPER_TYPES


def test_careful_edit_is_deterministic():
    a = _careful(seed_from=3)
    b = _careful(seed_from=3)
    assert a[0].file_ops == b[0].file_ops and a[2] == b[2]


@pytest.mark.parametrize("variant", ["anchored", "retypeset"])
def test_careful_edit_changes_values_and_keeps_the_arithmetic(variant, tmp_path):
    plan, base, pdf = _careful(variant=variant)
    assert plan.label.method == "rewrite" and plan.label.arithmetic_consistent
    op = plan.file_ops[0]
    with pdfplumber.open(io.BytesIO(pdf)) as doc:
        text = "".join(p.extract_text() or "" for p in doc.pages)
    number = re.search(r"\d[\d,]*\.\d{2}", op.new).group()
    assert number in text.replace(" ", "")
    ing = ingest(str(_write(tmp_path, pdf)))
    assert ing.closing_balance == ingest(str(_write(tmp_path, base.pdf, "base.pdf"))).closing_balance
    codes = {f.code for f in financial_logic.analyze(ing).findings}
    assert not codes & {"FL_BALANCE_BREAK", "FL_CLOSING_MISMATCH"}


@pytest.mark.parametrize("variant", ["anchored", "retypeset"])
def test_careful_edit_leaves_nothing_the_document_checks_can_see(variant, tmp_path):
    _, base, pdf = _careful(variant=variant)
    assert structure_forensics.analyze_bytes(pdf) == []
    before = {f.code for f in visual_forensics.analyze(ingest(str(_write(tmp_path, base.pdf, "b.pdf")))).findings}
    after = {f.code for f in visual_forensics.analyze(ingest(str(_write(tmp_path, pdf)))).findings}
    assert after == before


def test_retypeset_keeps_the_right_edge_of_the_column():
    plan, base, pdf = _careful(variant="retypeset")
    op = next(o for o in plan.file_ops if o.field == "amount")
    cell = base.row_map.cells[op.row_id][op.field]
    with pdfplumber.open(io.BytesIO(pdf)) as doc:
        words = doc.pages[cell.page - 1].extract_words()
    [w] = [w for w in words if abs(w["top"] - cell.top) < 2 and op.new.endswith(w["text"])]
    assert abs(w["x1"] - cell.x1) < 0.05


# ---- typography features (6b) --------------------------------------------------------

def test_genuine_statement_has_no_typographic_outliers():
    pdf = load_renderer("two_col_dmon").render(make_test_ledger(3, 5), DEFAULT_STYLE, 5).pdf
    f = statement_geometry.extract(pdf)
    assert f["spacing_irregular_tokens"] == 0 and f["column_misaligned_tokens"] == 0


def test_retypeset_number_stands_out():
    _, _, pdf = _careful(variant="retypeset")
    assert statement_geometry.extract(pdf)["spacing_irregular_tokens"] > 0


def test_statement_typeset_unevenly_throughout_is_not_penalised():
    # The hard negative: uneven spacing everywhere is a typesetting habit, not an edit.
    pdf = load_renderer("generator_v1").render(make_test_ledger(3, 5), DEFAULT_STYLE, 5).pdf
    kerned = residual_eval.kern_everything(pdf, random.Random(1))
    assert kerned != pdf
    assert statement_geometry.extract(kerned)["spacing_irregular_tokens"] == 0


def test_number_ending_off_its_column_is_measured():
    renderer = load_renderer("generator_v1")
    ledger = make_test_ledger(3, 5)
    base = renderer.render(ledger, DEFAULT_STYLE, 5)
    row_id = next(r for r, cells in base.row_map.cells.items()
                  if re.search(r"\d,\d{3}\.\d{2}", cells["amount"].text))
    text = base.row_map.cells[row_id]["amount"].text
    longer = re.sub(r"(\d),(\d{3}\.\d{2})", r"1\1,\2", text, count=1)   # one more digit
    from fraud_detect.corpus.schema import FileOp
    op = FileOp(op="rewrite", row_id=row_id, field="amount", new=longer,
                params=(("variant", "anchored"),))
    pdf = apply_file_ops(base.pdf, (op,), base.row_map, 5)
    f = statement_geometry.extract(pdf)
    assert f["column_misaligned_tokens"] >= 1 and f["column_misalignment_max"] > 0.3


def test_unreadable_pdf_has_no_typography_features():
    assert statement_geometry.extract(b"%PDF-1.4 not a pdf") is None


# ---- behaviour features (6a) ------------------------------------------------------------

def _behaviour(tmp_path, name, seed):
    renderer = load_renderer("generator_v1")
    ledger = make_test_ledger(3, seed)
    genuine = renderer.render(ledger, DEFAULT_STYLE, seed).pdf
    plan = CATALOG[name].plan(ledger, renderer, rng(seed, "step3-test", name))
    base = renderer.render(plan.ledger, DEFAULT_STYLE, seed)
    tampered = apply_file_ops(base.pdf, plan.file_ops, base.row_map, seed)
    g = statement_behaviour.extract(ingest(str(_write(tmp_path, genuine, "g.pdf"))).transactions)
    t = statement_behaviour.extract(ingest(str(_write(tmp_path, tampered, "t.pdf"))).transactions)
    return g, t


def test_turnover_inflation_raises_receipts_against_spending(tmp_path):
    g, t = _behaviour(tmp_path, "turnover_inflation", 4)
    assert t["credit_debit_ratio"] > g["credit_debit_ratio"]


def test_fabricated_receipts_raise_receipts_against_spending(tmp_path):
    g, t = _behaviour(tmp_path, "fabricated_receipts", 4)
    assert t["credit_debit_ratio"] > g["credit_debit_ratio"]


def test_too_few_transactions_give_no_behaviour_features():
    assert statement_behaviour.extract([]) is None


# ---- the experiment's data plan ---------------------------------------------------------

def test_jobs_keep_splits_and_the_held_out_layout_apart():
    jobs = residual_eval.plan_jobs(3, 2, 2)
    seeds = {}
    for j in jobs:
        seeds.setdefault(j.split, set()).add(j.seed)
        assert (j.layout == residual_eval.HELD_OUT_LAYOUT) == (j.split == "test_layout")
    splits = list(seeds)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            assert not seeds[a] & seeds[b]
    assert {j.kind for j in jobs} >= {"genuine", "careful_edit", *residual_eval.HARD_NEGATIVES}
