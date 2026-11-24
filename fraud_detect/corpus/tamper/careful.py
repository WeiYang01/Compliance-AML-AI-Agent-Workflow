"""Careful in-place edit: ``careful_edit`` (an Extra_Tamper_Type, opt-in like ``subtle_amount_edit``).

What a careful forger does with a PDF editor: change a value by rewriting the page's own
text, with no overlay, no white box and no appended save, and keep the arithmetic right.

The plan moves money between two receipts of the same month: the earlier receipt goes up
by ``d`` and a later one goes down by ``d``. The balances between them change; the month's
totals and the closing balance do not, so every arithmetic check still passes. Each
changed cell is rewritten in place (``FileOp(op="rewrite")``) in one of two variants,
drawn from the tamper random stream:

* ``anchored``  - the new text keeps the original start position. In a right-aligned
  column a value of a different width then ends a little left or right of its column.
* ``retypeset`` - the right edge is kept, but the characters are re-spaced with small
  uneven gaps, the way an editor re-typesets one word.

Both leave only typographic traces. Neither leaves anything the overlay, hidden-text or
appended-save checks can see, which is the point: this is the residual case for Node 6b.
"""
from __future__ import annotations

import io
import random

from pypdf import PdfReader, PdfWriter
from pypdf.generic import (ArrayObject, ContentStream, FloatObject, NameObject,
                           TextStringObject)
from reportlab.pdfbase.pdfmetrics import stringWidth

from fraud_detect.corpus.schema import EditOp, FileOp, Ledger, RowMap, StyleKnobs, TamperLabel, TamperResult
from fraud_detect.corpus.tamper import ledger_ops
from fraud_detect.corpus.tamper._registry import info_for, register

NAME = "careful_edit"
VARIANTS = ("anchored", "retypeset")
LOW_PCT, HIGH_PCT = 5.0, 30.0       # d as a share of the smaller receipt
KERN_RANGE = (8, 45)                # retypeset: |gap| per glyph, thousandths of an em
_RIGHT_FIELDS = {"amount", "balance"}
_X_TOLERANCE = 0.75                 # points between a cell's x0 and its text position


def _credit_pairs(ledger: Ledger) -> list[tuple[int, int]]:
    """(earlier, later) indexes of credit rows in the same month."""
    pairs = []
    rows = ledger.rows
    for i, a in enumerate(rows):
        if a.amount <= 0:
            continue
        for j in range(i + 1, len(rows)):
            b = rows[j]
            if b.month_index != a.month_index:
                break
            if b.amount > 0:
                pairs.append((i, j))
    return pairs


@register(NAME)
def plan_careful_edit(ledger: Ledger, renderer, rng: random.Random) -> TamperResult:
    pairs = _credit_pairs(ledger)
    if not pairs:
        raise ValueError("careful_edit: no month has two receipts to move money between")
    i, j = pairs[rng.randrange(len(pairs))]
    a, b = ledger.rows[i], ledger.rows[j]
    d = round(min(a.amount, b.amount) * rng.uniform(LOW_PCT, HIGH_PCT) / 100.0, 2)
    if d <= 0:
        raise ValueError(f"careful_edit: receipts {a.row_id}/{b.row_id} are too small to change")
    variant = rng.choice(VARIANTS)
    edits = [EditOp("set_amount", a.row_id, "amount", ledger_ops._money(a.amount),
                    ledger_ops._money(a.amount + d)),
             EditOp("set_amount", b.row_id, "amount", ledger_ops._money(b.amount),
                    ledger_ops._money(b.amount - d)),
             EditOp("recompute_balances")]
    edited = ledger_ops.apply_edits(ledger, edits)

    genuine_cells = renderer.render(ledger, StyleKnobs(), 0).row_map.cells
    edited_cells = renderer.render(edited, StyleKnobs(), 0).row_map.cells
    file_ops = []
    for row_id, fields in genuine_cells.items():
        for field, cell in fields.items():
            new = edited_cells[row_id][field].text
            if new == cell.text:
                continue
            params = [("variant", variant)]
            if variant == "retypeset":
                kern = [rng.randint(*KERN_RANGE) * rng.choice((-1, 1)) for _ in new[:-1]]
                params.append(("kern", ",".join(map(str, kern))))
            file_ops.append(FileOp(op="rewrite", row_id=row_id, field=field, new=new,
                                   params=tuple(params)))

    info = info_for(NAME)
    aff = (ledger_ops._affected(ledger, a.row_id, "amount", edits[0].original, edits[0].new),
           ledger_ops._affected(ledger, b.row_id, "amount", edits[1].original, edits[1].new))
    label = TamperLabel(
        is_tampered=True, tamper_types=(NAME,), method="rewrite",
        arithmetic_consistent=ledger_ops.is_arithmetic_consistent(edited),
        affected_rows=aff, affected_months=(aff[0]["month"],),
        should_be_seen_by=info.should_be_seen_by,
        notes=f"{variant}: {len(file_ops)} cell(s) rewritten in place; totals unchanged.")
    return TamperResult(ledger=ledger, edits=(), file_ops=tuple(file_ops), label=label)


# ---------------------------------------------------------------------------
# applying the rewrite
# ---------------------------------------------------------------------------
def _fonts(page) -> dict[str, str]:
    fonts = page.get("/Resources", {}).get("/Font", {})
    return {str(k): str(v.get_object().get("/BaseFont", "/Helvetica")).lstrip("/")
            for k, v in fonts.items()}


def _rewrite_page(reader: PdfReader, page_no: int, ops: list[tuple[FileOp, object]]) -> ContentStream:
    page = reader.pages[page_no - 1]
    fonts = _fonts(page)
    content = ContentStream(page.get_contents(), reader)
    todo = list(ops)
    font, size = "Helvetica", 8.0
    tm_index = None
    for k, (operands, operator) in enumerate(content.operations):
        if operator == b"Tf":
            font, size = fonts.get(str(operands[0]), "Helvetica"), float(operands[1])
        elif operator == b"Tm":
            tm_index = k
        elif operator == b"Tj" and tm_index is not None and todo:
            text = str(operands[0])
            x = float(content.operations[tm_index][0][4])
            for n, (op, cell) in enumerate(todo):
                if text == cell.text and abs(x - cell.x0) <= _X_TOLERANCE:
                    params = dict(op.params)
                    right = x + stringWidth(text, font, size)
                    if params.get("variant") == "retypeset":
                        kern = [int(v) for v in params["kern"].split(",")] if params.get("kern") else []
                        items: list = []
                        for c, glyph in enumerate(op.new):
                            items.append(TextStringObject(glyph))
                            if c < len(kern):
                                items.append(FloatObject(kern[c]))
                        width = stringWidth(op.new, font, size) - sum(kern) / 1000.0 * size
                        content.operations[k] = ([ArrayObject(items)], b"TJ")
                    else:
                        content.operations[k] = ([TextStringObject(op.new)], b"Tj")
                        width = stringWidth(op.new, font, size)
                    if params.get("variant") == "retypeset" and op.field in _RIGHT_FIELDS:
                        tm = list(content.operations[tm_index][0])
                        tm[4] = FloatObject(round(right - width, 4))
                        content.operations[tm_index] = (tm, b"Tm")
                    del todo[n]
                    break
    if todo:
        missing = ", ".join(f"{op.row_id}.{op.field}" for op, _ in todo)
        raise ValueError(f"careful_edit: could not find the text of {missing} on page {page_no}")
    return content


def apply_rewrites(pdf: bytes, ops: list[FileOp], row_map: RowMap) -> bytes:
    """Rewrite each cell's text in place in the page content (see module docstring)."""
    reader = PdfReader(io.BytesIO(pdf))
    by_page: dict[int, list[tuple[FileOp, object]]] = {}
    for op in ops:
        cell = row_map.cells[op.row_id][op.field]
        by_page.setdefault(cell.page, []).append((op, cell))
    writer = PdfWriter(clone_from=reader)
    meta = dict(reader.metadata or {})
    for page_no, items in sorted(by_page.items()):
        content = _rewrite_page(reader, page_no, items)
        page = writer.pages[page_no - 1]
        page[NameObject("/Contents")] = writer._add_object(content)
    if meta:
        writer.add_metadata({str(k): str(v) for k, v in meta.items()})
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()
