"""Typographic features for Node 6b: traces a careful in-place edit leaves.

The rule-based checks (Node 2 and its structural checks) see fonts, formats, metadata,
text drawn over text, hidden text and appended saves. A value rewritten in place leaves
none of those. What it can leave is typography that does not match the rest of the
document:

* **Uneven spacing inside a number.** An editor that re-typesets one word places each
  character itself; the rest of the document was set in one go. Measured RELATIVE to the
  document's own spacing, so a statement typeset with kerning throughout is not penalised.
* **A money value out of line with its column.** Amount and balance columns are
  right-aligned by the bank's software; an edited value that keeps its old start position
  can end a little left or right of the column. Dates are left-aligned, so their right
  edges vary naturally and are not measured.

Only numeric tokens (amounts, balances, dates) are used. Nothing here is a rule: each
value is a feature for the model, and none duplicates a rule-based check.
"""
from __future__ import annotations

import io
import re
import statistics
from typing import Optional

FEATURE_NAMES = [
    "spacing_irregularity_max",    # worst token's spacing, above the document's typical spacing
    "spacing_irregular_tokens",    # tokens whose spacing stands out from the document
    "column_misaligned_tokens",    # numbers ending just off their column's right edge
    "column_misalignment_max",     # worst such offset, in font sizes
]

_NUMERIC = re.compile(r"\d[\d,./-]*\d")
#: Money values (amounts, balances) are right-aligned; dates and references are not.
_MONEY = re.compile(r"\d\.\d{2}\b")
_ALIGNED = 0.5        # points: a number this close to a column edge is aligned
_NEAR = 12.0          # points: further than this from any column edge is another column
_MIN_COLUMN = 4       # numbers that must share an edge to make a column
MIN_SPREAD = 0.005    # font sizes: the smallest spacing spread a document is given


def _token_spacing(chars: list) -> Optional[float]:
    """Mean absolute gap between consecutive characters, in font sizes."""
    if len(chars) < 3:
        return None
    size = max(statistics.median(c["size"] for c in chars), 1.0)
    gaps = [chars[k + 1]["x0"] - chars[k]["x1"] for k in range(len(chars) - 1)]
    return statistics.fmean(abs(g) for g in gaps) / size


def _page_tokens(page) -> list[tuple[str, list]]:
    words = page.extract_words(keep_blank_chars=False, use_text_flow=False,
                               return_chars=True, extra_attrs=["size"])
    return [(w["text"], w["chars"]) for w in words if _NUMERIC.search(w["text"])]


def _misalignment(tokens: list[tuple[float, float, float]]) -> list[float]:
    """Offsets (in font sizes) of money values that sit just off a right-aligned column.

    ``tokens`` are ``(right edge, top, size)``. A column is a right edge shared by at
    least ``_MIN_COLUMN`` values; a value is out of line only when it lies within the
    column's own vertical span (its first to last row, plus one row's spacing at each end).
    Summary boxes above or below the table are often aligned to a slightly different edge
    and are left alone.
    """
    columns: list[tuple[float, float, float]] = []      # (edge, first top, last top)
    for x, _, _ in sorted(tokens):
        if any(abs(x - c[0]) <= _ALIGNED for c in columns):
            continue
        members = [t for t in tokens if abs(t[0] - x) <= _ALIGNED]
        if len(members) >= _MIN_COLUMN:
            tops = sorted(t[1] for t in members)
            pitch = statistics.median(b - a for a, b in zip(tops, tops[1:]))
            # One row's leeway at each end, so the first and last rows count too.
            columns.append((x, tops[0] - pitch, tops[-1] + pitch))
    offsets = []
    for x, top, size in tokens:
        spans = [abs(x - edge) for edge, first, last in columns if first <= top <= last]
        nearest = min(spans, default=None)
        if nearest is not None and _ALIGNED < nearest <= _NEAR:
            offsets.append(nearest / max(size, 1.0))
    return offsets


def extract(pdf_bytes: bytes) -> Optional[dict[str, float]]:
    """Typographic features of one PDF, or ``None`` if it cannot be read."""
    import pdfplumber

    try:
        spacings: list[float] = []
        offsets: list[float] = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as doc:
            for page in doc.pages:
                tokens = _page_tokens(page)
                for _, chars in tokens:
                    s = _token_spacing(chars)
                    if s is not None:
                        spacings.append(s)
                offsets += _misalignment([(chars[-1]["x1"], chars[-1]["top"], chars[-1]["size"])
                                          for text, chars in tokens if _MONEY.search(text)])
    except Exception:  # noqa: BLE001 - an unreadable file has no features
        return None
    if not spacings:
        return {name: 0.0 for name in FEATURE_NAMES}
    typical = statistics.median(spacings)
    # A floor on the spread: on a document set in one go every gap is 0, so its own
    # spread is only rounding noise and would make any tiny difference look huge.
    spread = max(statistics.median(abs(s - typical) for s in spacings), MIN_SPREAD)
    excess = [(s - typical) / spread for s in spacings]
    return {
        "spacing_irregularity_max": max(0.0, max(excess)),
        "spacing_irregular_tokens": float(sum(1 for e in excess if e > 6.0)),
        "column_misaligned_tokens": float(len(offsets)),
        "column_misalignment_max": max(offsets, default=0.0),
    }
