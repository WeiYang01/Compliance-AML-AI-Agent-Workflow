"""Like-for-like Node 6a comparison: the row model against the statement model.

An offline analysis, outside the governed commands: it releases nothing, records no
decision and loosens no target. It answers one question the ``compare`` command cannot
when the row model fails its cutoff target: on the same documents, does the statement
feature set beat the row feature set?

For each training seed, both models are fit on that seed's train split with the
training code ``train`` uses (``train_a._restricted_pipeline`` on the audit-v1 6a
columns; ``train_s.make_model`` on the audit-v2 statement columns). Each model's cutoffs
are fixed on that seed's validation split at a 5% and a 1% false-alarm rate. Both are
then scored on the same documents: the seed's test split, its unseen-layout split and
the held-out seed corpus.

Two populations are reported:
  core  genuine statements plus tampered statements with at least one fraud row (the
        population both ``train`` paths use);
  all   core plus tampered statements whose transactions are untouched, as positives.

All numbers come from synthetic data and do not measure real-world accuracy.

    python -m fraud_detect.ml.retraining.like_for_like_6a [--output-root DIR] [--cache-dir DIR]

Needs the corpora from ``build``; uses the row-table cache that ``train`` writes. It only
prints; it writes no file.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from fraud_detect.ml import transaction_features as tf
from fraud_detect.ml.statement_behaviour import stage_b_features

from .audit import AUDIT_V1, AUDIT_V2, STATEMENT_NODE, model_input_columns
from .corpora import load_corpus
from .features_a import load_row_table
from .features_s import records_from_rows
from .train_a import _restricted_pipeline, statement_scores
from .train_s import make_model

FA_TARGETS = {"fa5": 0.05, "fa1": 0.01}
METRICS = ("roc_auc", "pr_auc", "catch@fa5", "fa@fa5", "catch@fa1", "fa@fa1")


def doc_table(corpus_dir, cache_root):
    """Per scorable document: label, split, population, tamper types, statement features."""
    corpus = load_corpus(corpus_dir)
    rt = load_row_table(corpus_dir, cache_root=cache_root)
    rows = rt.rows
    fraud = {k for k, v in rows.groupby(["corpus_key", "doc_id"])["y"].max().items() if v == 1}
    grouped = {k: g for k, g in rows.groupby(["corpus_key", "doc_id"], sort=False)}
    docs = {}
    for d in corpus.docs:
        key = (d.corpus_key, d.doc_id)
        if key not in grouped:
            continue
        sf = stage_b_features(records_from_rows(grouped[key]))
        if sf is None:  # too short for the statement features: neither model is compared on it
            continue
        docs[key] = {"split": d.split, "y": int(bool(d.is_tampered)),
                     "pop": "core" if (not d.is_tampered or key in fraud) else "untouched",
                     "sf": sf, "tamper": list(d.tamper_types)}
    return docs, rt.features


def fit_row(features):
    cols = set(model_input_columns(AUDIT_V1, "6a"))
    cols = [c for c in tf.PDF_FEATURE_COLUMNS if c in cols]
    cat = [c for c in tf.PDF_CATEGORICAL if c in cols]
    num = [c for c in cols if c not in cat]
    train = features[features["split"] == "train"]
    pipe = _restricted_pipeline(cat, num).fit(train[cols], train["y"].astype(int))
    return lambda f: statement_scores(pipe, f, cols)


def fit_statement(docs):
    cols = model_input_columns(AUDIT_V2, STATEMENT_NODE)
    train = [v for v in docs.values() if v["split"] == "train" and v["pop"] == "core"]
    model = make_model().fit(np.array([[v["sf"][c] for c in cols] for v in train]),
                             np.array([v["y"] for v in train]))

    def score(dd):
        keys = list(dd)
        p = model.predict_proba(np.array([[dd[k]["sf"][c] for c in cols] for k in keys]))[:, 1]
        return dict(zip(keys, map(float, p)))
    return score


def cutoff_at(scores, labels, fa) -> float:
    """Lowest cutoff whose false-alarm rate on these genuine scores is at most ``fa``."""
    g = np.sort(np.asarray([s for s, y in zip(scores, labels) if y == 0], dtype=float))
    k = int(np.floor(fa * len(g)))
    return float(np.nextafter(g[len(g) - k - 1], np.inf))


def metrics(scores, labels, cuts) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    s, y = np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)
    out = {"n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
           "roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s))}
    for name, c in cuts.items():
        alert = s >= c
        out[f"catch@{name}"] = float(alert[y == 1].mean())
        out[f"fa@{name}"] = float(alert[y == 0].mean())
    return out


def compare_seed(seed_dir, held_dir, cache_root) -> dict:
    docs, feats = doc_table(seed_dir, cache_root)
    hdocs, hfeats = doc_table(held_dir, cache_root)
    row_model, st_model = fit_row(feats), fit_statement(docs)

    def both(dd, ff):
        rs, ss = row_model(ff), st_model(dd)
        return {k: (rs[k], ss[k]) for k in dd if k in rs}

    scores, hscores = both(docs, feats), both(hdocs, hfeats)
    val = [k for k, v in docs.items() if v["split"] == "validation" and v["pop"] == "core" and k in scores]
    yv = [docs[k]["y"] for k in val]
    cuts = {m: {n: cutoff_at([scores[k][i] for k in val], yv, fa) for n, fa in FA_TARGETS.items()}
            for i, m in enumerate(("row", "statement"))}
    sets = {"test": ({k: v for k, v in docs.items() if v["split"] == "test"}, scores),
            "unseen_layout": ({k: v for k, v in docs.items() if v["split"] == "test_unseen_layout"}, scores),
            "heldout_seed": (hdocs, hscores)}
    result = {"cutoffs": cuts, "eval": {}, "docs": {}}
    for name, (dd, table) in sets.items():
        keys = [k for k in dd if k in table]
        for pop in ("core", "all"):
            ks = [k for k in keys if pop == "all" or dd[k]["pop"] == "core"]
            y = [dd[k]["y"] for k in ks]
            for i, m in enumerate(("row", "statement")):
                result["eval"][f"{name}/{pop}/{m}"] = metrics([table[k][i] for k in ks], y, cuts[m])
        result["docs"][name] = [{"doc_id": k[1], "y": dd[k]["y"], "pop": dd[k]["pop"],
                                 "tamper": dd[k]["tamper"], "row": table[k][0],
                                 "statement": table[k][1]} for k in keys]
    return result


def summarise(per_seed: dict) -> str:
    seeds = sorted(per_seed)
    lines = ["Node 6a like-for-like: row model vs statement model "
             "(synthetic data, not real-world accuracy)", f"training seeds: {seeds}"]
    for key in per_seed[seeds[0]]["eval"]:
        name, pop, model = key.split("/")
        if model != "row":
            continue
        first = per_seed[seeds[0]]["eval"][key]
        lines.append(f"\n{name} / {pop} ({first['n_pos']} tampered / {first['n_neg']} genuine per seed)")
        for met in METRICS:
            r = np.array([per_seed[s]["eval"][f"{name}/{pop}/row"][met] for s in seeds])
            t = np.array([per_seed[s]["eval"][f"{name}/{pop}/statement"][met] for s in seeds])
            lines.append(f"  {met:10s} row {r.mean():.3f}  statement {t.mean():.3f}  "
                         f"difference {(t - r).mean():+.3f}  (statement higher in {int((t > r).sum())}/{len(seeds)})")
    # who catches what on the held-out seed at the 5% cutoffs
    per, fa_or = defaultdict(lambda: np.zeros(4)), []
    for s in seeds:
        c = per_seed[s]["cutoffs"]
        for d in per_seed[s]["docs"]["heldout_seed"]:
            ra, sa = d["row"] >= c["row"]["fa5"], d["statement"] >= c["statement"]["fa5"]
            if not d["y"]:
                fa_or.append(ra or sa)
                continue
            for t in d["tamper"]:
                per[(t, d["pop"])] += [1, ra, sa, ra or sa]
    lines.append(f"\nheld-out seed, 5% cutoffs, catch rate per tamper type "
                 f"(either = flagged by at least one; its false-alarm rate {np.mean(fa_or):.3f})")
    for (t, pop), v in sorted(per.items()):
        lines.append(f"  {t:24s} {pop:9s} row {v[1] / v[0]:.2f}  statement {v[2] / v[0]:.2f}  either {v[3] / v[0]:.2f}")
    return "\n".join(lines)


def main(argv=None) -> int:
    from .config import RunConfig
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output-root", default="data/private/ml_retraining")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args(argv)
    root = Path(args.output_root)
    cache = Path(args.cache_dir) if args.cache_dir else root / "cache"
    cfg = RunConfig()
    held = root / "corpora" / f"heldout-seed-{cfg.held_out_seed}"
    per_seed = {s: compare_seed(root / "corpora" / f"seed-{s}", held, cache) for s in cfg.seeds}
    print(summarise(per_seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
