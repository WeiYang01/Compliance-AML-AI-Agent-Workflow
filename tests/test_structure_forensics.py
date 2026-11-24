"""Node 2 structural forensics (fraud_detect/nodes/structure_forensics.py).

Clean statements and tampers that do not touch the file's structure must raise
nothing; each structural trace must raise exactly one finding that names what
was found.
"""
from __future__ import annotations

import io
import re

import pytest
from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas

from fraud_detect.corpus.schema import LAYOUTS
from fraud_detect.corpus.seeding import rng
from fraud_detect.corpus.tamper.catalog import CATALOG
from fraud_detect.corpus.tamper.pdf_ops import apply_file_ops
from fraud_detect.models import Routing, Severity
from fraud_detect.nodes import structure_forensics as sf
from fraud_detect.nodes.visual_forensics import AGENT as NODE2
from fraud_detect.pipeline import run_pipeline
from fraud_detect.synthetic import generate_clean
from tests.corpus.helpers import DEFAULT_STYLE, load_renderer, make_test_ledger

STRUCTURAL = {"VF_TEXT_OVER_TEXT", "VF_TEXT_COVERED", "VF_INCREMENTAL_EDIT",
              "VF_STRUCTURE_UNREADABLE"}


def _codes(findings):
    return sorted(f.code for f in findings)


def _tampered(name, layout="generator_v1", *, method=None, background=None):
    """First seed whose plan for ``name`` matches the requested method/background."""
    renderer = load_renderer(layout)
    for seed in range(40):
        plan = CATALOG[name].plan(make_test_ledger(2, seed), renderer, rng(seed, "sf-test", name))
        if method and plan.label.method != method:
            continue
        if background is not None and not any(o.background == background for o in plan.file_ops):
            continue
        base = renderer.render(plan.ledger, DEFAULT_STYLE, seed)
        return plan, base, apply_file_ops(base.pdf, plan.file_ops, base.row_map, seed)
    pytest.skip(f"no seed gives {name} with method={method} background={background}")


def _clean_pdf(layout="generator_v1", seed=4):
    return load_renderer(layout).render(make_test_ledger(2, seed), DEFAULT_STYLE, seed).pdf


def _merge(pdf: bytes, draw) -> bytes:
    """Add a layer drawn by ``draw(canvas)`` on top of page 1."""
    reader = PdfReader(io.BytesIO(pdf))
    writer = PdfWriter(clone_from=reader)
    page = writer.pages[0]
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=(float(page.mediabox.width), float(page.mediabox.height)))
    draw(c)
    c.showPage()
    c.save()
    page.merge_page(PdfReader(io.BytesIO(buf.getvalue())).pages[0])
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def _append_revision(pdf: bytes, extra: bytes = b"") -> bytes:
    """Append an incremental update (a new Info dictionary), as Acrobat-style editors do."""
    prev = int(re.findall(rb"startxref\s+(\d+)", pdf)[-1])
    root = re.search(rb"/Root\s+(\d+\s+\d+\s+R)", pdf).group(1)
    size = int(re.findall(rb"/Size\s+(\d+)", pdf)[-1])
    body = pdf if pdf.endswith(b"\n") else pdf + b"\n"
    offset = len(body)
    obj = b"%d 0 obj\n<< /Producer (Adobe Acrobat Pro) %s >>\nendobj\n" % (size, extra)
    xref_at = offset + len(obj)
    xref = (b"xref\n%d 1\n%010d 00000 n \ntrailer\n<< /Size %d /Root %s /Info %d 0 R /Prev %d >>\n"
            b"startxref\n%d\n%%%%EOF\n" % (size, offset, size + 1, root, size, prev, xref_at))
    return body + obj + xref


# ---- nothing to report ---------------------------------------------------------

@pytest.mark.parametrize("layout", LAYOUTS)
def test_clean_statements_raise_nothing(layout):
    # Includes layouts whose header lines are so tight that word boxes touch.
    assert sf.analyze_bytes(_clean_pdf(layout)) == []


@pytest.mark.parametrize("name", ["genuine_resaved", "metadata_inconsistency",
                                  "structuring", "fabricated_receipts"])
def test_tampers_that_leave_the_structure_alone_raise_nothing(name):
    _, _, pdf = _tampered(name)
    assert sf.analyze_bytes(pdf) == []


def test_an_added_layer_away_from_the_text_is_not_reported():
    # Stamps, page numbers and watermarks are often added as a separate layer.
    def stamp(c):
        c.setFont("Helvetica", 7)
        c.drawString(40, 12, "Downloaded from online banking")
    assert sf.analyze_bytes(_merge(_clean_pdf(), stamp)) == []


# ---- text drawn over text ----------------------------------------------------------

@pytest.mark.parametrize("name", ["edited_amount", "subtle_amount_edit", "altered_date",
                                  "broken_balance"])
def test_pasted_over_value_is_reported_with_the_original_underneath(name):
    plan, base, pdf = _tampered(name, method="overlay")
    [f] = [f for f in sf.analyze_bytes(pdf) if f.code == "VF_TEXT_OVER_TEXT"]
    assert f.severity == Severity.HIGH and f.agent == NODE2
    op = next(o for o in plan.file_ops if o.op in ("overlay", "set_font"))
    original = base.row_map.cells[op.row_id][op.field].text
    assert "drawn over" in f.evidence
    assert op.new.split()[-1] in f.evidence and original.split()[-1] in f.evidence


def test_white_box_over_the_original_is_reported():
    _, _, pdf = _tampered("edited_amount", method="overlay", background=True)
    codes = _codes(sf.analyze_bytes(pdf))
    assert codes == ["VF_TEXT_COVERED", "VF_TEXT_OVER_TEXT"]


def test_value_drawn_exactly_on_top_in_one_layer_is_reported():
    # The careful forger's case: same spot, same layer. Word extraction merges the
    # two values into one garbled word, so the check must work on characters.
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=A4)
    c.setFont("Helvetica", 9)
    c.drawString(400, 700, "12,500.00")
    c.drawString(400, 700, "92,500.00")
    c.drawString(72, 650, "Closing balance")
    c.showPage()
    c.save()
    [f] = sf.analyze_bytes(buf.getvalue())
    assert f.code == "VF_TEXT_OVER_TEXT"
    assert "'92,500.00' drawn over '12,500.00'" in f.evidence


# ---- appended edits ----------------------------------------------------------------

def test_changes_appended_after_saving_are_reported():
    pdf = _append_revision(_clean_pdf())
    assert PdfReader(io.BytesIO(pdf)).metadata.producer == "Adobe Acrobat Pro"  # a valid update
    [f] = sf.analyze_bytes(pdf)
    assert f.code == "VF_INCREMENTAL_EDIT" and "saved 2 times" in f.evidence


def test_signed_files_are_not_reported_for_their_signature_revision():
    pdf = _append_revision(_clean_pdf(), extra=b"/ByteRange [0 0 0 0]")
    assert sf.analyze_bytes(pdf) == []


def test_linearised_files_are_not_counted_as_edited():
    pdf = _clean_pdf()
    head = pdf.index(b"\n", pdf.index(b"%PDF")) + 1
    pdf = pdf[:head] + b"% /Linearized 1\nstartxref\n0\n" + pdf[head:]
    assert sf.analyze_bytes(pdf) == []


# ---- fail safe and wiring ------------------------------------------------------------

def test_unreadable_file_is_reported_not_passed():
    [f] = sf.analyze_bytes(b"%PDF-1.4 this is not a pdf")
    assert f.code == "VF_STRUCTURE_UNREADABLE" and f.severity == Severity.LOW


def test_pipeline_reports_structural_findings_under_node_2(tmp_path):
    clean = run_pipeline(generate_clean(str(tmp_path)).path)
    assert clean.routing == Routing.AUTO_CLEAR
    assert not STRUCTURAL & {f.code for f in clean.findings}

    _, _, pdf = _tampered("subtle_amount_edit", method="overlay")
    path = tmp_path / "overlay.pdf"
    path.write_bytes(pdf)
    report = run_pipeline(str(path))
    [f] = [f for f in report.findings if f.code == "VF_TEXT_OVER_TEXT"]
    assert f.agent == NODE2 and report.routing == Routing.HUMAN_REVIEW
