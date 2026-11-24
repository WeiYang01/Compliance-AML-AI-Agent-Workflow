"""Node 1 validation layer (fraud_detect/nodes/extraction_check.py).

It checks Node 1's READING, not the document's honesty: clean and tampered
statements read correctly must raise nothing, and each kind of misread must
raise exactly one data-quality finding without changing any parsed value.
"""
from __future__ import annotations

import os

import pytest

from fraud_detect.corpus.schema import LAYOUTS
from fraud_detect.models import DocumentMetadata, Routing, Severity
from fraud_detect.nodes import extraction_check as ec
from fraud_detect.nodes import risk_synthesis
from fraud_detect.pipeline import run_pipeline
from fraud_detect.store import CaseStore
from fraud_detect.synthetic import generate_clean
from tests.corpus.helpers import render_and_ingest

CODES = {"DQ_EXTRACTOR_DISAGREE", "DQ_VALUE_NOT_IN_TEXT", "DQ_METADATA_UNREAD"}


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    from fraud_detect.nodes.ingestion import ingest
    gen = generate_clean(str(tmp_path_factory.mktemp("clean")))
    return ingest(gen.path)


def _same_pages(ing):
    return lambda _path: [p.text for p in ing.page_texts]


# ---- clean documents ----------------------------------------------------------

@pytest.mark.parametrize("layout", LAYOUTS)
def test_clean_statements_raise_nothing_and_every_field_is_verified(layout):
    rendered = render_and_ingest(layout, 2, 11)  # keep it: it owns the temporary PDF
    ing = rendered.ingestion
    check, findings = ec.validate(ing)
    assert findings == []
    assert check.second_reader_available and check.pages_disagreeing == []
    assert check.pages_compared == len(ing.page_texts)
    assert check.rows_checked > 0 and check.rows_not_found == 0
    assert {f.status for f in check.fields} <= {"verified", "missing"}


@pytest.mark.parametrize("tamper", ["edited_amount", "broken_balance", "metadata_inconsistency"])
def test_tampering_is_not_reported_as_a_reading_problem(tamper, tmp_path):
    # Reading checks and fraud checks must stay separate: a tampered file that
    # was read correctly raises no validation finding (Nodes 2/3 judge it).
    from fraud_detect.corpus.seeding import rng
    from fraud_detect.corpus.tamper.catalog import CATALOG
    from fraud_detect.corpus.tamper.pdf_ops import apply_file_ops
    from fraud_detect.nodes.ingestion import ingest
    from tests.corpus.helpers import DEFAULT_STYLE, load_renderer, make_test_ledger

    renderer = load_renderer("generator_v1")
    plan = CATALOG[tamper].plan(make_test_ledger(2, 3), renderer, rng(3, "ec-test", tamper))
    base = renderer.render(plan.ledger, DEFAULT_STYLE, 3)
    path = tmp_path / f"{tamper}.pdf"
    path.write_bytes(apply_file_ops(base.pdf, plan.file_ops, base.row_map, 3))
    _, findings = ec.validate(ingest(str(path)))
    assert findings == []


# ---- each kind of misread -------------------------------------------------------

def test_value_not_in_text_is_one_medium_finding_listing_the_values(clean):
    rows = [t.model_copy(update={"amount": 987654.32}) if i < 7 else t
            for i, t in enumerate(clean.transactions)]
    ing = clean.model_copy(update={"closing_balance": 123456.78, "transactions": rows})
    check, findings = ec.validate(ing, second_reader=_same_pages(ing))
    [f] = findings
    assert f.code == "DQ_VALUE_NOT_IN_TEXT" and f.severity == Severity.MEDIUM
    assert "closing_balance 123,456.78" in f.evidence and "and 3 more" in f.evidence
    assert check.rows_not_found == 7
    assert {c.name: c.status for c in check.fields}["closing_balance"] == "not_found_in_text"


def test_misread_header_text_field_is_not_found(clean):
    ing = clean.model_copy(update={"account_holder": "SOMEONE ELSE PTE LTD"})
    check, findings = ec.validate(ing, second_reader=_same_pages(ing))
    assert [f.code for f in findings] == ["DQ_VALUE_NOT_IN_TEXT"]
    assert findings[0].field == "account_holder"


def test_misread_value_sends_the_case_to_a_human(clean):
    # Fail safe: a screening built on numbers the document does not contain is
    # never auto-cleared.
    ing = clean.model_copy(update={"closing_balance": 123456.78})
    _, findings = ec.validate(ing, second_reader=_same_pages(ing))
    ing.data_quality_findings.extend(findings)
    report = risk_synthesis.synthesize(ing, [])
    assert report.routing == Routing.HUMAN_REVIEW


def test_readers_disagreeing_is_one_low_finding_naming_the_page(clean):
    pages = [p.text for p in clean.page_texts]
    pages[-1] = pages[-1] + "\n 4,444.44"
    _, findings = ec.validate(clean, second_reader=lambda _p: pages)
    [f] = findings
    assert f.code == "DQ_EXTRACTOR_DISAGREE" and f.severity == Severity.LOW
    assert f.page == clean.page_texts[-1].page and "4444.44" in f.evidence


def test_page_count_mismatch_counts_as_disagreement(clean):
    pages = [p.text for p in clean.page_texts] + ["extra page 1.00"]
    check, findings = ec.validate(clean, second_reader=lambda _p: pages)
    assert [f.code for f in findings] == ["DQ_EXTRACTOR_DISAGREE"]
    assert check.pages_disagreeing


def test_unreadable_second_reader_is_skipped_not_assumed_to_agree(clean):
    def broken(_path):
        raise OSError("cannot open")
    check, findings = ec.validate(clean, second_reader=broken)
    assert findings == []
    assert check.second_reader_available is False and check.pages_compared == 0


def test_metadata_in_file_but_not_read_is_reported(clean):
    ing = clean.model_copy(update={"metadata": DocumentMetadata()})
    _, findings = ec.validate(ing, second_reader=_same_pages(ing),
                              info_reader=lambda _p: {"/Producer": "x", "/Title": "y"})
    assert [f.code for f in findings] == ["DQ_METADATA_UNREAD"]


def test_file_without_metadata_is_not_a_reading_problem(clean):
    ing = clean.model_copy(update={"metadata": DocumentMetadata()})
    check, findings = ec.validate(ing, second_reader=_same_pages(ing), info_reader=lambda _p: {})
    assert findings == [] and check.metadata_entries_in_file == 0


def test_missing_fields_are_marked_missing_without_a_new_finding(clean):
    # Node 1 already raises DQ_MISSING_* for these; do not count them twice.
    ing = clean.model_copy(update={"uen": None, "opening_balance": None})
    check, findings = ec.validate(ing, second_reader=_same_pages(ing))
    assert findings == []
    statuses = {c.name: c.status for c in check.fields}
    assert statuses["uen"] == statuses["opening_balance"] == "missing"


def test_every_misread_at_once_gives_at_most_one_finding_per_check(clean):
    rows = [t.model_copy(update={"balance": 1.11 * (i + 1000)}) for i, t in enumerate(clean.transactions)]
    ing = clean.model_copy(update={"transactions": rows, "metadata": DocumentMetadata()})
    _, findings = ec.validate(ing, second_reader=lambda _p: ["9.99"] * len(ing.page_texts),
                              info_reader=lambda _p: {"/Producer": "x"})
    assert sorted(f.code for f in findings) == sorted(CODES)


def test_validation_never_changes_a_parsed_value(clean):
    ing = clean.model_copy(update={"closing_balance": 123456.78}, deep=True)
    before = ing.model_dump()
    ec.validate(ing, second_reader=lambda _p: ["9.99"] * len(ing.page_texts))
    assert ing.model_dump() == before


# ---- wiring -----------------------------------------------------------------

def test_pipeline_attaches_the_check_and_it_survives_the_case_store(tmp_path):
    gen = generate_clean(str(tmp_path))
    report = run_pipeline(gen.path)
    assert report.extraction_check is not None
    assert report.extraction_check.rows_checked > 0
    assert not CODES & {f.code for f in report.findings}

    store = CaseStore(os.path.join(tmp_path, "cases.json"),
                      audit_path=os.path.join(tmp_path, "audit.jsonl"))
    store.upsert_report(report)
    stored = CaseStore(os.path.join(tmp_path, "cases.json"),
                       audit_path=os.path.join(tmp_path, "audit.jsonl")).get(report.case_id)
    assert stored.report.extraction_check == report.extraction_check
