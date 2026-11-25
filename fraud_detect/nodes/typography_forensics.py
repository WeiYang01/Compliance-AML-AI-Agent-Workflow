"""Node 2 typography checks: traces a careful in-place edit leaves.

The structural checks (``structure_forensics``) catch text drawn over text, hidden text
and appended saves. A value rewritten in place, in the page's own text, leaves none of
those. What it can leave is typography that does not match the rest of the document:

1. **A number spaced differently from the rest of its document**
   (``VF_SPACING_OUTLIER``). An editor that re-typesets one value places each character
   itself; the rest of the document was set in one go. Spacing is measured RELATIVE to
   the document's own spacing, so a statement typeset unevenly throughout is not flagged.
2. **A money value just off its column** (``VF_COLUMN_MISALIGNED``). Amount and balance
   columns are right-aligned by the bank's software; an edited value that keeps its old
   start position can end a little left or right of the column. Only values within the
   column's own rows count: summary boxes above or below the table are often aligned to
   a slightly different edge. Dates are left-aligned and are not measured.

Evidence for the rule (``python -m fraud_detect.ml.residual_eval``, synthetic data): it
catches 25 of 29 re-typeset ``careful_edit`` values that every other check misses, with
no false alarm on 120 genuine statements (plain, re-saved, and typeset unevenly
throughout) or on any other tamper type. A trained model on the same evidence caught
29 of 29 but flagged about a fifth of genuine statements, so the evidence is used as a
rule. Both findings are MEDIUM: strong on synthetic data, unmeasured on real statements.

The same measurements are the Node 6b typography features
(``fraud_detect.ml.statement_geometry``), so the rule and the features cannot drift apart.
"""
from __future__ import annotations

import io
import re
import statistics
from dataclasses import dataclass, field
from typing import Optional

from ..models import Finding, Severity
from .visual_forensics import AGENT

_NUMERIC = re.compile(r"\d[\d,./-]*\d")
#: Money values (amounts, balances) are right-aligned; dates and references are not.
_MONEY = re.compile(r"\d\.\d{2}\b")
_ALIGNED = 0.5        # points: a number this close to a column edge is aligned
_NEAR = 12.0          # points: further than this from any column edge is another column
_MIN_COLUMN = 4       # numbers that must share an edge to make a column
#: Font sizes: the smallest spacing spread a document is given. On a document set in one
#: go every gap is 0, so its own spread is only rounding noise.
MIN_SPREAD = 0.005
#: A number stands out when its spacing is this many spreads above the document's typical.
SPACING_OUTLIER = 6.0
_MAX_EXAMPLES = 3


@dataclass(frozen=True)
class _Token:
    page: int
    text: str
    value: float          # spacing excess (in spreads) or column offset (in font sizes)


@dataclass
class Measurement:
    """Typographic measurements of one document."""
    spacing: list[tuple[int, str, float]] = field(default_factory=list)   # (page, text, spacing)
    misaligned: list[_Token] = field(default_factory=list)

    def spacing_excess(self) -> list[_Token]:
        """Every measured number with how far its spacing stands out (in spreads)."""
        if not self.spacing:
            return []
        values = [s for _, _, s in self.spacing]
        typical = statistics.median(values)
        spread = max(statistics.median(abs(s - typical) for s in values), MIN_SPREAD)
        return [_Token(p, t, (s - typical) / spread) for p, t, s in self.spacing]

    def outliers(self) -> list[_Token]:
        return [t for t in self.spacing_excess() if t.value > SPACING_OUTLIER]


def _token_spacing(chars: list) -> Optional[float]:
    """Mean absolute gap between consecutive characters, in font sizes."""
    if len(chars) < 3:
        return None
    size = max(statistics.median(c["size"] for c in chars), 1.0)
    gaps = [chars[k + 1]["x0"] - chars[k]["x1"] for k in range(len(chars) - 1)]
    return statistics.fmean(abs(g) for g in gaps) / size


def _misaligned(page_no: int, tokens: list[tuple[str, list]]) -> list[_Token]:
    """Money values that sit just off a right-aligned column, within its own rows."""
    points = [(text, chars[-1]["x1"], chars[-1]["top"], chars[-1]["size"])
              for text, chars in tokens if _MONEY.search(text)]
    columns: list[tuple[float, float, float]] = []      # (edge, first top, last top)
    for _, x, _, _ in sorted(points, key=lambda p: p[1]):
        if any(abs(x - c[0]) <= _ALIGNED for c in columns):
            continue
        tops = sorted(p[2] for p in points if abs(p[1] - x) <= _ALIGNED)
        if len(tops) >= _MIN_COLUMN:
            pitch = statistics.median(b - a for a, b in zip(tops, tops[1:]))
            # One row's leeway at each end, so the first and last rows count too.
            columns.append((x, tops[0] - pitch, tops[-1] + pitch))
    found = []
    for text, x, top, size in points:
        nearest = min((abs(x - edge) for edge, first, last in columns if first <= top <= last),
                      default=None)
        if nearest is not None and _ALIGNED < nearest <= _NEAR:
            found.append(_Token(page_no, text, nearest / max(size, 1.0)))
    return found


def measure(pdf_bytes: bytes) -> Optional[Measurement]:
    """Typographic measurements of one PDF, or ``None`` if it cannot be read."""
    import pdfplumber

    m = Measurement()
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as doc:
            for page_no, page in enumerate(doc.pages, 1):
                words = page.extract_words(keep_blank_chars=False, use_text_flow=False,
                                           return_chars=True, extra_attrs=["size"])
                tokens = [(w["text"], w["chars"]) for w in words if _NUMERIC.search(w["text"])]
                for text, chars in tokens:
                    s = _token_spacing(chars)
                    if s is not None:
                        m.spacing.append((page_no, text, s))
                m.misaligned += _misaligned(page_no, tokens)
    except Exception:  # noqa: BLE001 - structure_forensics reports an unreadable file
        return None
    return m


def _examples(tokens: list[_Token]) -> str:
    shown = "; ".join(f"page {t.page}: '{t.text}'" for t in tokens[:_MAX_EXAMPLES])
    extra = len(tokens) - _MAX_EXAMPLES
    return shown + (f"; and {extra} more" if extra > 0 else "")


def analyze_bytes(pdf_bytes: bytes) -> list[Finding]:
    """Typography findings for one PDF (see the module docstring)."""
    m = measure(pdf_bytes)
    if m is None:
        return []
    findings = []
    outliers = m.outliers()
    if outliers:
        findings.append(Finding(
            agent=AGENT, code="VF_SPACING_OUTLIER",
            title="Number spaced differently from the rest of the document",
            severity=Severity.MEDIUM, weight=12, page=outliers[0].page,
            evidence=(f"{len(outliers)} number(s) have character spacing unlike the rest of "
                      f"the statement ({_examples(outliers)}). The rest of the document was "
                      f"set in one go; a value re-typed in a PDF editor is often spaced by the "
                      f"editor instead. Compare these values with the bank's own records."),
        ))
    if m.misaligned:
        findings.append(Finding(
            agent=AGENT, code="VF_COLUMN_MISALIGNED",
            title="Money value out of line with its column",
            severity=Severity.MEDIUM, weight=10, page=m.misaligned[0].page,
            evidence=(f"{len(m.misaligned)} money value(s) end just off the right edge their "
                      f"column shares with the other rows ({_examples(m.misaligned)}). Bank "
                      f"software aligns these columns exactly; a value edited in place that "
                      f"kept its old start position ends slightly out of line."),
        ))
    return findings


def analyze(source_path: str) -> list[Finding]:
    """Typography findings for the PDF at ``source_path``."""
    try:
        with open(source_path, "rb") as fh:
            return analyze_bytes(fh.read())
    except OSError:
        return []
