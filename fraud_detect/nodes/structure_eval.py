"""Evaluation of Node 2 structural forensics on the synthetic corpus.

Run as ``python -m fraud_detect.nodes.structure_eval [--seeds N] [--months M]``.

Read-only: statements are generated into a temporary folder that is removed
afterwards; nothing in the repo is written. For every tamper type in the corpus
catalogue and every layout, it asks two questions:

* **Caught before**: did the existing checks (Node 2's visual checks or Node 3's
  financial checks) raise a MEDIUM-or-higher finding that does not also fire on
  the clean statements? Findings that fire on clean statements as well (e.g. the
  fictional companies are not in the registry) are background, not detection.
* **Caught by structure**: did the structural checks raise anything?

It also counts structural findings on clean statements (false alarms). The
corpus is synthetic, so these numbers show the checks separate the generator's
tampering methods; they are not a measurement on real statements.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import TextIO

_SERIOUS = ("MEDIUM", "HIGH")


def _style():
    from fraud_detect.corpus.schema import StyleKnobs

    return StyleKnobs(bank_name="NORTHERN TEST BANK",
                      address_lines=("1 Specimen Road", "Singapore 000000"),
                      producer="ReportLab PDF Library", creation_time="09:30:00",
                      row_step=12.4, second_address_line=False)


def _ledger(months: int, seed: int):
    from datetime import date

    from fraud_detect.corpus.company import make_company
    from fraud_detect.corpus.ledger import simulate_ledger
    from fraud_detect.corpus.schema import SimulatorKnobs

    return simulate_ledger(make_company(seed, "C0001"), date(2026, 1, 1), months, seed,
                           SimulatorKnobs(), None)


def _screen(path: Path) -> tuple[set[str], set[str]]:
    """(codes from the existing Node 2/3 checks at MEDIUM+, structural codes)."""
    from fraud_detect.nodes import financial_logic, structure_forensics, visual_forensics
    from fraud_detect.nodes.ingestion import ingest

    ing = ingest(str(path))
    before = {f.code for r in (visual_forensics.analyze(ing), financial_logic.analyze(ing))
              for f in r.findings if f.severity.value in _SERIOUS}
    return before, {f.code for f in structure_forensics.analyze(str(path))}


def run(seeds: int = 3, months: int = 3, out: TextIO = sys.stdout) -> dict:
    from fraud_detect.corpus.layouts import get_renderer
    from fraud_detect.corpus.schema import LAYOUTS
    from fraud_detect.corpus.seeding import rng
    from fraud_detect.corpus.tamper.catalog import CATALOG
    from fraud_detect.corpus.tamper.pdf_ops import apply_file_ops

    style = _style()
    rows: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with tempfile.TemporaryDirectory(prefix="structure_eval_") as tmp:
        path = Path(tmp) / "doc.pdf"
        background: set[str] = set()
        for layout in LAYOUTS:
            renderer = get_renderer(layout)
            for seed in range(seeds):
                path.write_bytes(renderer.render(_ledger(months, seed), style, seed).pdf)
                before, structural = _screen(path)
                background |= before
                rows["(clean)"]["docs"] += 1
                rows["(clean)"]["structure"] += bool(structural)
        for name in sorted(CATALOG):
            for layout in LAYOUTS:
                renderer = get_renderer(layout)
                for seed in range(seeds):
                    try:
                        plan = CATALOG[name].plan(_ledger(months, seed), renderer,
                                                  rng(seed, "structure-eval", name))
                    except ValueError:
                        continue   # the tamper does not fit this statement
                    base = renderer.render(plan.ledger, style, seed)
                    path.write_bytes(apply_file_ops(base.pdf, plan.file_ops, base.row_map, seed))
                    before, structural = _screen(path)
                    caught_before = bool(before - background)
                    row = rows[f"{name} [{plan.label.method}]"]
                    row["docs"] += 1
                    row["before"] += caught_before
                    row["structure"] += bool(structural)
                    row["after"] += caught_before or bool(structural)
                    row["only_structure"] += bool(structural) and not caught_before

    print(f"{'tamper [method]':40} {'docs':>5} {'before':>7} {'struct':>7} {'after':>6} {'new':>5}",
          file=out)
    for key in sorted(rows):
        r = rows[key]
        print(f"{key:40} {r['docs']:5} {r['before']:7} {r['structure']:7} {r['after']:6} "
              f"{r['only_structure']:5}", file=out)
    print(f"\nbackground codes ignored (fire on clean statements too): {sorted(background)}",
          file=out)
    return {k: dict(v) for k, v in rows.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--months", type=int, default=3)
    args = ap.parse_args(argv)
    run(args.seeds, args.months)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
