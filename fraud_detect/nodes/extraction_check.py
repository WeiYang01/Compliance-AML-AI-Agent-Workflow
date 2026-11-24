"""Node 1 validation layer: did we read the document correctly?

Runs straight after ingestion and checks Node 1's OWN reading, not the
document's honesty. It answers three questions:

1. **Do two independent readers agree?** The pages are read a second time with
   pypdf (Node 1 reads them with pdfplumber) and the money values on each page
   are compared. A disagreement means the text layer is hard to read (unusual
   font encoding, ligatures, broken character maps), so values parsed from that
   page are less certain.
2. **Is every parsed value really in the document?** Each header field and each
   transaction amount and balance must appear in the extracted text. A value
   that does not appear was produced by the parser, not read from the page.
3. **Was the file's metadata read?** Node 1 swallows metadata read errors and
   returns empty metadata, which would silently blind Node 2's editor checks.
   The metadata dictionary is read again independently and compared.

Design rules:
    * It never changes a parsed value. "Correcting" a number could hide the very
      edit Nodes 2 and 3 are looking for.
    * It never reports fraud. Every finding is a data-quality (``DQ_*``) finding
      from Node 1, so Node 4 treats it as reduced certainty.
    * At most one finding per check, so one badly read file cannot flood the
      confidence score.
    * Every check is fail-safe: if the second reader cannot open the file, the
      check is skipped and recorded as not compared rather than guessed.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Optional, Sequence

from ..models import ExtractionCheck, FieldCheck, Finding, IngestionResult, Severity
from .ingestion import AGENT  # findings belong to Node 1, like its other DQ findings

# A money value as printed: 1,234.56 or 1234.56 (sign, currency and CR/DR
# markers are ignored; values are compared without commas).
_MONEY_RE = re.compile(r"\d{1,3}(?:,\d{3})+\.\d{2}|\d+\.\d{2}")

# How many examples a finding lists before saying "and N more".
_MAX_EXAMPLES = 5

#: Header fields checked against the text, in display order.
TEXT_FIELDS = ("account_holder", "uen", "account_number")
MONEY_FIELDS = ("opening_balance", "closing_balance")

SecondReader = Callable[[str], Sequence[str]]
InfoReader = Callable[[str], dict]


def _money_tokens(text: str) -> Counter:
    return Counter(m.replace(",", "") for m in _MONEY_RE.findall(text or ""))


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text or "").lower()


def _pypdf_pages(path: str) -> list[str]:
    from pypdf import PdfReader

    return [page.extract_text() or "" for page in PdfReader(path).pages]


def _pypdf_info(path: str) -> dict:
    from pypdf import PdfReader

    info = PdfReader(path).metadata or {}
    return {k: v for k, v in dict(info).items() if v not in (None, "")}


def _examples(items: Sequence[str]) -> str:
    shown = ", ".join(items[:_MAX_EXAMPLES])
    extra = len(items) - _MAX_EXAMPLES
    return shown + (f" and {extra} more" if extra > 0 else "")


def _check_fields(ing: IngestionResult, money_in_text: Counter) -> tuple[list[FieldCheck], list[str]]:
    checks: list[FieldCheck] = []
    not_found: list[str] = []
    text = _squash(ing.raw_text)
    for name in TEXT_FIELDS:
        value = getattr(ing, name)
        if not value:
            checks.append(FieldCheck(name=name, status="missing"))
        elif _squash(value) in text:
            checks.append(FieldCheck(name=name, value=value, status="verified"))
        else:
            checks.append(FieldCheck(name=name, value=value, status="not_found_in_text"))
            not_found.append(f"{name} {value!r}")
    for name in MONEY_FIELDS:
        value = getattr(ing, name)
        if value is None:
            checks.append(FieldCheck(name=name, status="missing"))
        elif f"{abs(value):.2f}" in money_in_text:
            checks.append(FieldCheck(name=name, value=f"{value:.2f}", status="verified"))
        else:
            checks.append(FieldCheck(name=name, value=f"{value:.2f}", status="not_found_in_text"))
            not_found.append(f"{name} {value:,.2f}")
    return checks, not_found


def _check_rows(ing: IngestionResult, money_in_text: Counter) -> tuple[int, list[str]]:
    checked = 0
    not_found: list[str] = []
    for t in ing.transactions:
        for label, value in (("amount", t.amount), ("balance", t.balance)):
            if value is None:
                continue
            checked += 1
            if f"{abs(value):.2f}" not in money_in_text:
                not_found.append(f"row {t.index} {label} {value:,.2f}")
    return checked, not_found


def validate(ing: IngestionResult, *, second_reader: Optional[SecondReader] = None,
             info_reader: Optional[InfoReader] = None) -> tuple[ExtractionCheck, list[Finding]]:
    """Check how far Node 1's reading of ``ing`` can be confirmed.

    ``second_reader`` and ``info_reader`` default to pypdf on ``ing.source_path``;
    tests pass stand-ins. Returns the per-field report and at most three
    ``DQ_*`` findings.
    """
    second_reader = second_reader or _pypdf_pages
    info_reader = info_reader or _pypdf_info
    findings: list[Finding] = []
    money_in_text = _money_tokens(ing.raw_text)

    # ---- 1. a second, independent reader -------------------------------------
    pages_compared = 0
    disagreeing: list[int] = []
    examples: list[str] = []
    reader_ok = True
    try:
        second = list(second_reader(ing.source_path))
    except Exception:  # noqa: BLE001 - an unreadable file is "not compared", never "agreed"
        second, reader_ok = [], False
    if reader_ok:
        if len(second) != len(ing.page_texts):
            disagreeing = [p.page for p in ing.page_texts] or [1]
            examples.append(f"page count {len(ing.page_texts)} vs {len(second)}")
        else:
            for page, other in zip(ing.page_texts, second):
                pages_compared += 1
                a, b = _money_tokens(page.text), _money_tokens(other)
                if a != b:
                    disagreeing.append(page.page)
                    only_first = sorted((a - b).elements())[:2]
                    only_second = sorted((b - a).elements())[:2]
                    examples.append(f"page {page.page}: {only_first} vs {only_second}")
    if disagreeing:
        findings.append(Finding(
            agent=AGENT, code="DQ_EXTRACTOR_DISAGREE",
            title="Two text readers disagree on the numbers",
            severity=Severity.LOW, weight=3, page=disagreeing[0],
            evidence=(f"The document was read twice, with two independent tools, and the "
                      f"money values differ on {len(disagreeing)} page(s) "
                      f"({_examples(examples)}). The text layer is hard to read on these "
                      f"pages, so values parsed from them are less certain. This is a reading "
                      f"problem, not evidence of tampering; check the figures against the PDF."),
        ))

    # ---- 2. every parsed value must appear in the text -----------------------
    fields, fields_missing_from_text = _check_fields(ing, money_in_text)
    rows_checked, rows_missing_from_text = _check_rows(ing, money_in_text)
    not_in_text = fields_missing_from_text + rows_missing_from_text
    if not_in_text:
        findings.append(Finding(
            agent=AGENT, code="DQ_VALUE_NOT_IN_TEXT",
            title="Parsed values not found in the document",
            severity=Severity.MEDIUM, weight=5,
            field=fields_missing_from_text[0].split()[0] if fields_missing_from_text else None,
            evidence=(f"{len(not_in_text)} value(s) Node 1 extracted do not appear anywhere in "
                      f"the document text: {_examples(not_in_text)}. The screening may be using "
                      f"numbers the statement does not contain, so a person should check these "
                      f"values against the PDF before relying on the result."),
        ))

    # ---- 3. was the file's metadata read? ------------------------------------
    try:
        info = info_reader(ing.source_path)
    except Exception:  # noqa: BLE001
        info = {}
    meta = ing.metadata
    read = sum(1 for v in (meta.author, meta.creator, meta.producer,
                           meta.creation_date, meta.mod_date, meta.title) if v)
    if info and read == 0:
        findings.append(Finding(
            agent=AGENT, code="DQ_METADATA_UNREAD",
            title="File metadata present but not read",
            severity=Severity.LOW, weight=2, field="metadata",
            evidence=(f"The file carries {len(info)} metadata entries (e.g. "
                      f"{_examples(sorted(info))}) but Node 1 read none of them, so the "
                      f"editing-software and date checks in Node 2 had nothing to examine."),
        ))

    check = ExtractionCheck(
        fields=fields,
        rows_checked=rows_checked,
        rows_not_found=len(rows_missing_from_text),
        pages_compared=pages_compared,
        pages_disagreeing=disagreeing,
        second_reader_available=reader_ok,
        metadata_entries_in_file=len(info),
        metadata_fields_read=read,
    )
    return check, findings
