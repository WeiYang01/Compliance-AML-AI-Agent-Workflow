"""Node 2 structural forensics: evidence inside the PDF file itself.

Node 2's original checks (``visual_forensics``) look at fonts, date and currency
formats, and metadata. These checks look at how the file was BUILT, which most
editing leaves behind even when the page looks perfect:

1. **Text drawn over text** (``VF_TEXT_OVER_TEXT``). A genuine statement never
   prints two different characters in the same place. Pasting a new value over an
   old one usually leaves the old value in the file, underneath. Characters keep
   the order they were drawn in, so the value drawn later is reported as the added
   one and the value under it as the likely original.
2. **Text hidden under a white box** (``VF_TEXT_COVERED``). A white rectangle drawn
   in an added layer over existing text, the usual way to hide the original.
3. **Changes appended after the file was finished** (``VF_INCREMENTAL_EDIT``).
   Editors such as Acrobat save changes by appending them to the end of the file,
   so the earlier version is still inside it.

Design rules:
    * An added layer on its own is not reported. Stamps, page numbers and
      watermarks are often added that way; only an added layer that sits on top of
      existing text is.
    * Two characters count as overlapping only when their boxes cover a real share
      of each other (40% of the height, 30% of the width); tightly spaced lines
      touch at the edges and must not count. Identical characters drawn on top of
      each other are ignored: some tools fake bold text by printing it twice.
    * A digitally signed file is not reported for appended changes: signing
      appends a revision by design.
    * If the file cannot be examined, a LOW finding says so; the checks are never
      silently treated as passed.

Findings carry Node 2's agent name: they are one more view of the same document,
not an independent agent, so they must not earn Node 4's corroboration bonus.
"""
from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass

from ..models import Finding, Severity
from .visual_forensics import AGENT

_log = logging.getLogger(__name__)

#: Minimum share of the smaller box two characters must cover to count as drawn
#: over each other (vertical, horizontal). Measured on the synthetic corpus: lower
#: values flag tightly spaced header lines; a pasted value covers far more.
MIN_VERTICAL_OVERLAP = 0.4
MIN_HORIZONTAL_OVERLAP = 0.3
#: A cover box must hide at least half of a word in both directions.
MIN_COVER = 0.5
#: Fill colours at or above this level (0-1, all channels) count as white.
WHITE_LEVEL = 0.95
_MAX_EXAMPLES = 3

_LINEARIZED = re.compile(rb"/Linearized\b")


@dataclass(frozen=True)
class _Box:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float


def _overlaps(a: _Box, b: _Box, vmin: float, hmin: float) -> bool:
    """True when the boxes cover at least ``vmin``/``hmin`` of the smaller one."""
    v = min(a.bottom, b.bottom) - max(a.top, b.top)
    h = min(a.x1, b.x1) - max(a.x0, b.x0)
    if v <= 0 or h <= 0:
        return False
    height = min(a.bottom - a.top, b.bottom - b.top) or 1.0
    width = min(a.x1 - a.x0, b.x1 - b.x0) or 1.0
    return v / height >= vmin and h / width >= hmin


@dataclass(frozen=True)
class _Overlap:
    page: int
    added: str      # the value drawn later (on top)
    under: str      # the value underneath


@dataclass
class _Run:
    """Characters drawn one after another along a line (one piece of printed text)."""
    chars: list

    def words_meeting(self, other: "_Run") -> str:
        """The words of this run whose span meets ``other``'s span on the line."""
        lo = min(c["x0"] for c in other.chars)
        hi = max(c["x1"] for c in other.chars)
        words, current = [], []
        for c in self.chars:
            if c["text"].isspace():
                if current:
                    words.append(current)
                current = []
            else:
                current.append(c)
        if current:
            words.append(current)
        hit = ["".join(ch["text"] for ch in w) for w in words
               if min(ch["x0"] for ch in w) < hi and max(ch["x1"] for ch in w) > lo]
        return " ".join(hit) or "".join(c["text"] for c in self.chars).strip()


def _runs(chars: list) -> tuple[list[_Run], dict[int, int]]:
    """Split characters (in drawing order) into runs; return runs and char -> run."""
    runs: list[_Run] = []
    owner: dict[int, int] = {}
    prev = None
    for i, ch in enumerate(chars):
        size = max(ch["bottom"] - ch["top"], 1.0)
        starts_new = (prev is None
                      or abs(ch["top"] - prev["top"]) > 0.5 * size
                      or ch["x0"] < prev["x0"] - 0.5
                      or ch["x0"] - prev["x1"] > 1.5 * size)
        if starts_new:
            runs.append(_Run([]))
        runs[-1].chars.append(ch)
        owner[i] = len(runs) - 1
        prev = ch
    return runs, owner


def _page_overlaps(page_no: int, chars: list) -> list[_Overlap]:
    """Pairs of runs where a later run draws a different character over an earlier one."""
    runs, owner = _runs(chars)
    grid: dict[tuple[int, int], list[int]] = {}
    for i, ch in enumerate(chars):
        grid.setdefault((int(ch["top"] // 6), int(ch["x0"] // 6)), []).append(i)
    pairs: set[tuple[int, int]] = set()
    for i, a in enumerate(chars):
        if a["text"].isspace():
            continue
        gy, gx = int(a["top"] // 6), int(a["x0"] // 6)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for j in grid.get((gy + dy, gx + dx), ()):
                    if j <= i or owner[j] == owner[i]:
                        continue
                    b = chars[j]
                    if b["text"].isspace() or b["text"] == a["text"]:
                        continue
                    if _overlaps(_Box("", a["x0"], a["x1"], a["top"], a["bottom"]),
                                 _Box("", b["x0"], b["x1"], b["top"], b["bottom"]),
                                 MIN_VERTICAL_OVERLAP, MIN_HORIZONTAL_OVERLAP):
                        pairs.add((owner[i], owner[j]))   # i drawn first, j on top
    found = []
    for under_i, added_i in sorted(pairs):
        under, added = runs[under_i], runs[added_i]
        found.append(_Overlap(page_no, added.words_meeting(under), under.words_meeting(added)))
    return found


def _is_white(colour) -> bool:
    if colour is None:
        return False
    values = colour if isinstance(colour, (list, tuple)) else [colour]
    try:
        return bool(values) and all(float(v) >= WHITE_LEVEL for v in values)
    except (TypeError, ValueError):
        return False


def _read(pdf_bytes: bytes) -> list[tuple[list[_Box], list[_Box], list]]:
    """Words, white filled boxes and characters (in drawing order) for every page."""
    import pdfplumber

    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as doc:
        for page in doc.pages:
            words = [_Box(w["text"], w["x0"], w["x1"], w["top"], w["bottom"])
                     for w in page.extract_words(keep_blank_chars=False, use_text_flow=False)]
            covers = [_Box("", r["x0"], r["x1"], r["top"], r["bottom"])
                      for r in page.rects
                      if r.get("fill") and _is_white(r.get("non_stroking_color"))]
            chars = [{k: c[k] for k in ("text", "x0", "x1", "top", "bottom")}
                     for c in page.chars]
            pages.append((words, covers, chars))
    return pages


def _layers(reader, index: int):
    """(earlier layers, last layer) of a page with several content streams, else None.

    Each is ``(words, white boxes)``. Used to tell a white box ADDED over the page
    from an ordinary cell background, which is drawn before the text it sits under.
    """
    from pypdf import PdfWriter
    from pypdf.generic import ArrayObject, NameObject

    contents = reader.pages[index].get("/Contents")
    contents = contents.get_object() if contents is not None else None
    if not isinstance(contents, ArrayObject) or len(contents) < 2:
        return None

    def only(pick):
        writer = PdfWriter()
        writer.add_page(reader.pages[index])
        page = writer.pages[0]
        page[NameObject("/Contents")] = ArrayObject(pick(list(page["/Contents"])))
        buf = io.BytesIO()
        writer.write(buf)
        words, covers, _ = _read(buf.getvalue())[0]
        return words, covers

    return only(lambda items: items[:-1]), only(lambda items: items[-1:])


def _page_covers(page_no: int, layers) -> list[tuple[int, str]]:
    if layers is None:
        return []
    (earlier, _), (_, covers) = layers
    return [(page_no, w.text) for c in covers for w in earlier
            if _overlaps(c, w, MIN_COVER, MIN_COVER)]


def _revisions(pdf_bytes: bytes) -> int:
    """How many times the file was saved (1 for a file written in one go)."""
    count = pdf_bytes.count(b"startxref")
    if _LINEARIZED.search(pdf_bytes[:2048]):
        count -= 1   # a linearised ("fast web view") file has two by design
    return max(1, count)


def _is_signed(pdf_bytes: bytes) -> bool:
    return b"/ByteRange" in pdf_bytes


def _examples(items: list[str]) -> str:
    extra = len(items) - _MAX_EXAMPLES
    return "; ".join(items[:_MAX_EXAMPLES]) + (f"; and {extra} more" if extra > 0 else "")


def analyze_bytes(pdf_bytes: bytes) -> list[Finding]:
    """Structural findings for one PDF (see the module docstring)."""
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = _read(pdf_bytes)
        overlaps: list[_Overlap] = []
        covers: list[tuple[int, str]] = []
        for i, (_, _, chars) in enumerate(pages):
            overlaps += _page_overlaps(i + 1, chars)
            covers += _page_covers(i + 1, _layers(reader, i))
    except Exception:  # noqa: BLE001 - a file we cannot examine is reported, not passed
        _log.warning("structural forensics could not examine the file", exc_info=True)
        return [Finding(
            agent=AGENT, code="VF_STRUCTURE_UNREADABLE",
            title="File structure could not be examined",
            severity=Severity.LOW, weight=3,
            evidence=("The PDF's internal structure could not be read, so the checks for "
                      "text drawn over text, hidden text and appended edits did not run. "
                      "This is not evidence of tampering; a person should open the file."),
        )]

    findings: list[Finding] = []
    if overlaps:
        items = [f"page {o.page}: '{o.added}' drawn over '{o.under}'" for o in overlaps]
        findings.append(Finding(
            agent=AGENT, code="VF_TEXT_OVER_TEXT",
            title="Text drawn over other text (possible pasted-over value)",
            severity=Severity.HIGH, weight=25, page=overlaps[0].page,
            evidence=(f"{len(overlaps)} value(s) were drawn on top of existing text: "
                      f"{_examples(items)}. The value underneath is still in the file and is "
                      f"likely the original. A genuine statement never prints two values in "
                      f"the same place; this is the usual trace of a new value pasted over an "
                      f"old one."),
        ))
    if covers:
        items = [f"page {p}: '{t}'" for p, t in covers]
        findings.append(Finding(
            agent=AGENT, code="VF_TEXT_COVERED",
            title="Existing text hidden under a white box",
            severity=Severity.MEDIUM, weight=10, page=covers[0][0],
            evidence=(f"A white box drawn in an added layer covers {len(covers)} word(s) of "
                      f"the original page ({_examples(items)}). Covering text with a white "
                      f"box is a common way to hide a value before printing a new one."),
        ))
    revisions = _revisions(pdf_bytes)
    if revisions > 1 and not _is_signed(pdf_bytes):
        findings.append(Finding(
            agent=AGENT, code="VF_INCREMENTAL_EDIT",
            title="Changes appended after the file was finished",
            severity=Severity.MEDIUM, weight=10, field="file",
            evidence=(f"The file was saved {revisions} times, each later save appending "
                      f"changes to the end. Editors such as Acrobat save edits this way, "
                      f"and bank systems normally write a statement in one go. The earlier "
                      f"version is still inside the file and can be recovered for comparison."),
        ))
    return findings


def analyze(source_path: str) -> list[Finding]:
    """Structural findings for the PDF at ``source_path``."""
    try:
        with open(source_path, "rb") as fh:
            data = fh.read()
    except OSError:
        data = b""
    return analyze_bytes(data)
