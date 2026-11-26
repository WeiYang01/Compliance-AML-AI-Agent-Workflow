"""Step 3 experiment: can Node 6a / 6b catch what the rule-based checks miss?

Run as ``python -m fraud_detect.ml.residual_eval [--train 24 --val 16 --test 10 --workers 4]``.

Read-only: statements are generated into a temporary folder that is removed afterwards,
and no model file is written. It is a fast experiment, not the governed Stage B pipeline
(``fraud_detect/ml/retraining``); results that justify it move there in a later step.

Protocol
--------
* **Data.** Synthetic statements from the corpus generator. Train, validation and test
  use disjoint seeds (so disjoint companies); one layout (``two_col_ddmmyy_cr``) is held
  out entirely and only tested on. For every layout and seed: one genuine statement, one
  genuine hard negative (re-saved by a print/export tool, or typeset with uneven spacing
  throughout), one ``careful_edit``, one catalogue tamper in rotation, and one content or
  behaviour tamper in rotation (Node 6a's targets, so the residual has enough of them).
* **The residual.** For each document the rule-based checks run (Node 2 visual,
  structural and typography checks, Node 3). A tampered document is *missed* when they raise no
  MEDIUM-or-higher finding beyond the background codes that also fire on genuine
  training statements. The missed documents are the residual: the only place an ML node
  can add anything.
* **Models.** Gradient-boosted trees on four feature sets: Node 6b's current features,
  current plus the new typography features, the typography features alone, and the new
  statement-behaviour features for Node 6a. Each is trained on the training split with every tampered document as positive,
  and its threshold is set on the validation genuines for a 5% false-alarm rate.
* **Leak check.** Any single feature that separates the classes on validation with an
  AUC of 0.98 or more is listed: on generated data that usually means it reflects how the
  generator prints, not tampering.

Everything is synthetic. The numbers show whether the features carry information the
rules do not use on the generator's output; they are not real-world accuracy.
"""
from __future__ import annotations

import argparse
import io
import math
import random
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TextIO

HELD_OUT_LAYOUT = "two_col_ddmmyy_cr"
HARD_NEGATIVES = ("genuine_resaved", "genuine_kerned")
TARGET_FALSE_ALARM = 0.05
LEAK_AUC = 0.98
_SERIOUS = ("MEDIUM", "HIGH")
_SPLIT_SEED_BASE = {"train": 0, "validation": 500, "test": 1000, "test_layout": 2000}


@dataclass(frozen=True)
class Job:
    layout: str
    seed: int
    split: str
    kind: str          # "genuine", a hard negative, or a tamper type


# ---------------------------------------------------------------------------
# document generation (runs in worker processes)
# ---------------------------------------------------------------------------
def _style():
    from fraud_detect.corpus.schema import StyleKnobs

    return StyleKnobs(bank_name="NORTHERN TEST BANK",
                      address_lines=("1 Specimen Road", "Singapore 000000"),
                      producer="ReportLab PDF Library", creation_time="09:30:00",
                      row_step=12.4, second_address_line=False)


def _ledger(seed: int, months: int = 3):
    from datetime import date

    from fraud_detect.corpus.company import make_company
    from fraud_detect.corpus.ledger import simulate_ledger
    from fraud_detect.corpus.schema import SimulatorKnobs

    return simulate_ledger(make_company(seed, f"C{seed:05d}"), date(2026, 1, 1), months, seed,
                           SimulatorKnobs(), None)


def kern_everything(pdf: bytes, rng: random.Random) -> bytes:
    """A genuine statement typeset with uneven character spacing throughout.

    Every text run gets small per-character gaps that sum to zero, so widths and column
    alignment are unchanged. A model must therefore look for spacing that stands out
    from the document, not for spacing as such.
    """
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import ArrayObject, ContentStream, FloatObject, NameObject, TextStringObject

    reader = PdfReader(io.BytesIO(pdf))
    writer = PdfWriter(clone_from=reader)
    for n, page in enumerate(reader.pages):
        content = ContentStream(page.get_contents(), reader)
        for k, (operands, operator) in enumerate(content.operations):
            if operator != b"Tj":
                continue
            text = str(operands[0])
            if len(text) < 2:
                continue
            kern = [rng.randint(8, 45) * rng.choice((-1, 1)) for _ in text[:-1]]
            # Zero-sum, spread over every gap (same overall width, column alignment kept).
            # Putting the whole correction on the last gap would push a letter back over
            # its neighbour, which no typesetter does.
            mean = sum(kern) / len(kern)
            kern = [round(k - mean) for k in kern]
            kern[-1] -= sum(kern)                      # remove the rounding remainder (<= 1 per gap)
            items: list = []
            for c, glyph in enumerate(text):
                items.append(TextStringObject(glyph))
                if c < len(kern):
                    items.append(FloatObject(kern[c]))
            content.operations[k] = ([ArrayObject(items)], b"TJ")
        writer.pages[n][NameObject("/Contents")] = writer._add_object(content)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def _build(job: Job) -> Optional[tuple[bytes, str]]:
    from fraud_detect.corpus.layouts import get_renderer
    from fraud_detect.corpus.seeding import rng
    from fraud_detect.corpus.tamper.catalog import CATALOG
    from fraud_detect.corpus.tamper.pdf_ops import apply_file_ops

    renderer = get_renderer(job.layout)
    ledger = _ledger(job.seed)
    style = _style()
    if job.kind == "genuine":
        return renderer.render(ledger, style, job.seed).pdf, ""
    if job.kind == "genuine_kerned":
        pdf = renderer.render(ledger, style, job.seed).pdf
        return kern_everything(pdf, random.Random(job.seed * 7919 + 1)), ""
    try:
        plan = CATALOG[job.kind].plan(ledger, renderer, rng(job.seed, "residual-eval", job.kind))
    except ValueError:
        return None                                    # the tamper does not fit this statement
    base = renderer.render(plan.ledger, style, job.seed)
    variant = plan.label.notes.split(":", 1)[0] if job.kind == "careful_edit" else ""
    return apply_file_ops(base.pdf, plan.file_ops, base.row_map, job.seed), variant


def screen(job: Job) -> Optional[dict]:
    """Build one document and compute its rule findings and all three feature sets."""
    from fraud_detect.ml import statement_behaviour, statement_features, statement_geometry
    from fraud_detect.nodes import (financial_logic, structure_forensics,
                                    typography_forensics, visual_forensics)
    from fraud_detect.nodes.ingestion import ingest

    built = _build(job)
    if built is None:
        return None
    pdf, variant = built
    with tempfile.TemporaryDirectory(prefix="residual_eval_") as tmp:
        path = Path(tmp) / "doc.pdf"
        path.write_bytes(pdf)
        ing = ingest(str(path))
        findings = (visual_forensics.analyze(ing).findings
                    + structure_forensics.analyze(str(path))
                    + typography_forensics.analyze_bytes(pdf)
                    + financial_logic.analyze(ing).findings)
        current = statement_features.extract_features(str(path)) or {}
    return {
        "layout": job.layout, "seed": job.seed, "split": job.split,
        "kind": f"{job.kind} ({variant})" if variant else job.kind,
        "tampered": job.kind not in ("genuine",) + HARD_NEGATIVES,
        "codes": sorted({f.code for f in findings if f.severity.value in _SERIOUS}),
        "current_6b": {k: float(current.get(k, math.nan)) for k in statement_features.FEATURE_NAMES},
        "typography": statement_geometry.extract(pdf) or {},
        "behaviour": statement_behaviour.extract(ing.transactions) or {},
    }


def plan_jobs(n_train: int, n_val: int, n_test: int) -> list[Job]:
    from fraud_detect.corpus.schema import LAYOUTS
    from fraud_detect.corpus.tamper.catalog import CATALOG

    rotation = sorted(n for n in CATALOG if n not in HARD_NEGATIVES + ("careful_edit",))
    # Node 6a's targets: tampers that change content, not the file. A second rotating slot
    # draws only from these so the residual has enough of them to judge Node 6a.
    content = sorted(n for n in rotation if CATALOG[n].group in ("Content", "Behaviour"))
    counts = {"train": n_train, "validation": n_val, "test": n_test, "test_layout": n_test}
    jobs = []
    for li, layout in enumerate(LAYOUTS):
        splits = ("test_layout",) if layout == HELD_OUT_LAYOUT else ("train", "validation", "test")
        for split in splits:
            for k in range(counts[split]):
                seed = _SPLIT_SEED_BASE[split] + li * 100 + k
                jobs += [Job(layout, seed, split, "genuine"),
                         Job(layout, seed, split, HARD_NEGATIVES[k % 2]),
                         Job(layout, seed, split, "careful_edit"),
                         Job(layout, seed, split, rotation[(k + li) % len(rotation)]),
                         Job(layout, seed, split, content[(k + 2 * li) % len(content)])]
    return jobs


# ---------------------------------------------------------------------------
# evaluation (main process)
# ---------------------------------------------------------------------------
FEATURE_SETS = {
    "6b current": ("current_6b",),
    "6b current + typography": ("current_6b", "typography"),
    "6b typography only": ("typography",),
    "6a statement behaviour": ("behaviour",),
}


def _matrix(docs: list[dict], groups: tuple[str, ...]):
    import numpy as np

    names = [f"{g}:{k}" for g in groups
             for k in sorted(next((d[g] for d in docs if d[g]), {}))]
    rows = []
    for d in docs:
        row = []
        for g in groups:
            values = d[g] or {}
            row += [values.get(n.split(":", 1)[1], math.nan) for n in names if n.startswith(g + ":")]
        rows.append(row)
    return np.array(rows, dtype=float), names


def _auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else math.nan


def run(n_train: int = 24, n_val: int = 16, n_test: int = 10, workers: int = 4,
        out: TextIO = sys.stdout) -> dict:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier

    jobs = plan_jobs(n_train, n_val, n_test)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        docs = [d for d in pool.map(screen, jobs, chunksize=4) if d is not None]
    feature_names = next(d for d in docs if d["behaviour"])["behaviour"].keys()
    for d in docs:                     # statements too short for behaviour features
        d["behaviour"] = d["behaviour"] or {k: math.nan for k in feature_names}

    by_split = {s: [d for d in docs if d["split"] == s] for s in _SPLIT_SEED_BASE}
    # Background: codes that fire on plain genuine statements. Hard negatives are left
    # out, or a re-save's codes would hide real detections of metadata tampering.
    background = {c for d in by_split["train"] if d["kind"] == "genuine" for c in d["codes"]}
    for d in docs:
        d["rules_hit"] = bool(set(d["codes"]) - background)

    p = lambda *a: print(*a, file=out)  # noqa: E731
    p(f"documents: {len(docs)} ({', '.join(f'{s} {len(v)}' for s, v in by_split.items())}); "
      f"held-out layout: {HELD_OUT_LAYOUT}")
    p(f"background codes (fire on genuine training statements): {sorted(background)}\n")

    results: dict = {"models": {}}
    test_sets = {"test (held-out seeds)": by_split["test"],
                 f"test ({HELD_OUT_LAYOUT})": by_split["test_layout"]}
    for title, test in test_sets.items():
        gen = [d for d in test if not d["tampered"]]
        tam = [d for d in test if d["tampered"]]
        by_kind = {}
        for d in gen:
            by_kind.setdefault(d["kind"], [0, 0])
            by_kind[d["kind"]][0] += d["rules_hit"]
            by_kind[d["kind"]][1] += 1
        p(f"rules only, {title}: catch {sum(d['rules_hit'] for d in tam)}/{len(tam)}, "
          f"false alarms {sum(d['rules_hit'] for d in gen)}/{len(gen)} "
          f"({', '.join(f'{k} {a}/{b}' for k, (a, b) in sorted(by_kind.items()))})")
    p("")

    per_type: dict = {}
    header = (f"{'model':26} {'test set':28} {'AUC':>5} {'false al.':>9} {'catch':>9} "
              f"{'residual':>9} {'rules+model':>11}")
    p(header)
    for model_name, groups in FEATURE_SETS.items():
        Xtr, names = _matrix(by_split["train"], groups)
        ytr = np.array([d["tampered"] for d in by_split["train"]])
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.1,
                                             random_state=0).fit(Xtr, ytr)
        Xv, _ = _matrix(by_split["validation"], groups)
        yv = np.array([d["tampered"] for d in by_split["validation"]])
        sv = clf.predict_proba(Xv)[:, 1]
        threshold = float(np.quantile(sv[~yv], 1 - TARGET_FALSE_ALARM))
        results["models"][model_name] = {"threshold": threshold}

        leaks = []
        for c, name in enumerate(names):
            col = Xv[:, c]
            ok = ~np.isnan(col)
            if ok.sum() > 10 and len(set(yv[ok])) == 2:
                a = _auc(yv[ok], col[ok])
                if max(a, 1 - a) >= LEAK_AUC:
                    leaks.append(f"{name} ({max(a, 1 - a):.2f})")
        results["models"][model_name]["leak_suspects"] = leaks

        for title, test in test_sets.items():
            X, _ = _matrix(test, groups)
            y = np.array([d["tampered"] for d in test])
            s = clf.predict_proba(X)[:, 1]
            flag = s > threshold
            rules = np.array([d["rules_hit"] for d in test])
            resid = y & ~rules
            row = {
                "auc": _auc(y, s),
                "false_alarm": f"{int((flag & ~y).sum())}/{int((~y).sum())}",
                "catch": f"{int((flag & y).sum())}/{int(y.sum())}",
                "residual_catch": f"{int((flag & resid).sum())}/{int(resid.sum())}",
                "system_catch": f"{int(((flag | rules) & y).sum())}/{int(y.sum())}",
            }
            results["models"][model_name][title] = row
            p(f"{model_name:26} {title:28} {row['auc']:5.2f} {row['false_alarm']:>9} "
              f"{row['catch']:>9} {row['residual_catch']:>9} {row['system_catch']:>11}")
            kinds = sorted({d["kind"] for d in test if not d["tampered"]})
            row["false_alarm_by_kind"] = {
                k: f"{sum(1 for d, f in zip(test, flag) if f and d['kind'] == k)}/"
                   f"{sum(1 for d in test if d['kind'] == k)}" for k in kinds}
            p(f"{'':26} {'':28} false alarms by kind: "
              + ", ".join(f"{k} {v}" for k, v in row["false_alarm_by_kind"].items()))
            for d, f in zip(test, flag):
                if d["tampered"] and not d["rules_hit"]:
                    per_type.setdefault(d["kind"], {}).setdefault(model_name, [0, 0])
                    per_type[d["kind"]][model_name][0] += int(f)
                    per_type[d["kind"]][model_name][1] += 1

    p("\nresidual (missed by the rules) by tamper type, both test sets: caught / missed-by-rules")
    p(f"{'tamper':28} " + " ".join(f"{m:>24}" for m in FEATURE_SETS))
    for kind in sorted(per_type):
        cells = [per_type[kind].get(m, [0, 0]) for m in FEATURE_SETS]
        p(f"{kind:28} " + " ".join(f"{c[0]:>21}/{c[1]:<2}" for c in cells))
    results["residual_by_type"] = per_type

    p("\nleak check (single feature AUC >= 0.98 on validation):")
    for model_name in FEATURE_SETS:
        leaks = results["models"][model_name]["leak_suspects"]
        p(f"  {model_name}: {', '.join(leaks) if leaks else 'none'}")
    p("\nAll numbers are on synthetic data from the corpus generator, not real-world accuracy.")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", type=int, default=24, help="seeds per layout for training")
    ap.add_argument("--val", type=int, default=16, help="seeds per layout for validation")
    ap.add_argument("--test", type=int, default=10, help="seeds per layout for testing")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    run(args.train, args.val, args.test, args.workers)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
