#!/usr/bin/env python3
"""Re-pin entries of ``tests/ml_retraining/protected_hashes.json``, with a recorded reason.

The pinned SHA-256 map guards Nodes 1-4, ``registry.py`` and ``reference_data.py``
against silent change. This script is the only intended way to update it.

    # Status: which protected files no longer match their pin (exit 1 if any, or if
    # the re-pin log is inconsistent).
    python scripts/repin_protected_hashes.py

    # Re-pin named files. Every changed file must be named; there is no "all".
    python scripts/repin_protected_hashes.py \\
        --file fraud_detect/nodes/risk_synthesis.py \\
        --reason "392ce09 added three display-only RiskReport fields; golden outputs unchanged" \\
        [--dry-run]

For each named file the script:
  1. refuses if the file still matches its pin, or is not in the map (adding a file to
     the protected set is a separate decision, made by editing the map in review);
  2. prints the diff from the last commit whose content matches the pin, so the person
     re-pinning looks at what actually changed;
  3. for ``risk_synthesis.py`` (Node 4 scoring), refuses unless the Node 4 golden test and
     the pinned-constants test pass: a scoring change must first be made deliberate by
     updating those tests in review, not waved through by a hash update;
  4. appends an entry (file, old, new, reason, date, diff_base) to ``repin_log``.

``tests/ml_retraining/test_repin.py`` checks that every file's log entries chain from its
baseline to the current pin, so a hand edit of the map without a log entry fails CI.
A determined hand edit can still forge a log entry; that is what code review is for.
The script only makes the honest path the easy one and puts the reason in the diff.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PINS = REPO / "tests" / "ml_retraining" / "protected_hashes.json"
NODE4 = "fraud_detect/nodes/risk_synthesis.py"
NODE4_GUARD_TESTS = [
    "tests/ml_retraining/test_node4_golden.py",
    "tests/ml_retraining/test_hygiene.py::test_node4_constants_and_literals_equal_pinned",
]

MIN_REASON_CHARS = 40
MIN_REASON_WORDS = 6
#: Reasons that say nothing about why the change is safe.
_PLACEHOLDER = re.compile(
    r"^(re-?pin(ned)?|update[sd]?|refresh(ed)?|fix(e[sd])?|bump(ed)?|sync(ed)?)?\s*"
    r"(the\s+)?(pinned\s+)?(hash(es)?|pins?|tests?|ci)?\s*(after|for)?\s*(changes?)?\.?$",
    re.I)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def dump(data: dict) -> str:
    """Serialise exactly as the committed file is formatted."""
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def reason_problems(reason: str | None) -> list[str]:
    r = (reason or "").strip()
    out = []
    if len(r) < MIN_REASON_CHARS:
        out.append(f"reason is {len(r)} characters; at least {MIN_REASON_CHARS} required")
    if len(r.split()) < MIN_REASON_WORDS:
        out.append(f"reason has {len(r.split())} words; at least {MIN_REASON_WORDS} required")
    if _PLACEHOLDER.match(r):
        out.append("reason only says the hash was updated; say what changed and why it is safe")
    return out


def log_problems(data: dict) -> list[str]:
    """Every pinned file has a log chain: baseline (old=None) -> ... -> current pin."""
    pins = data.get("sha256", {})
    log = data.get("repin_log")
    if not isinstance(log, list):
        return ["repin_log is missing or not a list"]
    out = []
    by_file: dict[str, list[dict]] = {}
    for i, e in enumerate(log):
        missing = {"file", "old", "new", "reason", "date", "kind"} - set(e)
        if missing:
            out.append(f"repin_log[{i}] lacks {sorted(missing)}")
            continue
        if e["kind"] not in ("baseline", "repin"):
            out.append(f"repin_log[{i}] has unknown kind {e['kind']!r}")
        if not _HEX64.match(str(e["new"])):
            out.append(f"repin_log[{i}] new is not a sha256")
        if e["kind"] == "repin":
            if not _HEX64.match(str(e["old"])):
                out.append(f"repin_log[{i}] old is not a sha256")
            for p in reason_problems(e["reason"]):
                out.append(f"repin_log[{i}] ({e['file']}): {p}")
        elif e["old"] is not None:
            out.append(f"repin_log[{i}] is a baseline entry with an old hash")
        by_file.setdefault(e["file"], []).append(e)
    for f in sorted(set(by_file) - set(pins)):
        out.append(f"{f}: in repin_log but not in the sha256 map")
    for f in sorted(pins):
        chain = by_file.get(f)
        if not chain:
            out.append(f"{f}: pinned but has no repin_log entry")
            continue
        if chain[0]["kind"] != "baseline":
            out.append(f"{f}: first repin_log entry is not the baseline")
        for prev, cur in zip(chain, chain[1:]):
            if cur["kind"] != "repin":
                out.append(f"{f}: second baseline entry")
            elif cur["old"] != prev["new"]:
                out.append(f"{f}: repin_log entry {cur['date']} has old {cur['old'][:12]}… "
                           f"but the previous pin was {prev['new'][:12]}…")
        if chain[-1]["new"] != pins[f]:
            out.append(f"{f}: pinned {pins[f][:12]}… but the last repin_log entry says "
                       f"{chain[-1]['new'][:12]}… (the map was edited without the script)")
    return out


def apply_repin(data: dict, current: dict[str, str], reason: str, date: str,
                diff_bases: dict[str, str | None]) -> dict:
    """Return a copy of ``data`` with each file in ``current`` re-pinned and logged."""
    new = json.loads(json.dumps(data))
    for rel in sorted(current):
        new["repin_log"].append({
            "kind": "repin", "file": rel, "old": new["sha256"][rel], "new": current[rel],
            "reason": reason.strip(), "date": date, "diff_base": diff_bases.get(rel),
        })
        new["sha256"][rel] = current[rel]
    return new


def _git(*args: str, binary: bool = False, repo: Path = REPO):
    try:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=not binary)
    except OSError:  # git not installed
        return subprocess.CompletedProcess(args, 1, b"" if binary else "", "")


def find_pinned_commit(rel: str, digest: str, repo: Path = REPO) -> str | None:
    """Most recent commit whose content of ``rel`` hashes to ``digest``."""
    out = _git("log", "--format=%H", "--", rel, repo=repo)
    for commit in out.stdout.split() if out.returncode == 0 else []:
        blob = _git("show", f"{commit}:{rel}", binary=True, repo=repo)
        if blob.returncode == 0 and sha256_bytes(blob.stdout) == digest:
            return commit
    return None


def node4_guard_passes() -> bool:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        *NODE4_GUARD_TESTS], cwd=REPO)
    return r.returncode == 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--file", action="append", default=[], metavar="PATH",
                    help="repo-relative protected file to re-pin (repeat for several)")
    ap.add_argument("--reason", help="what changed and why it is safe (required with --file)")
    ap.add_argument("--dry-run", action="store_true", help="show what would change; write nothing")
    ap.add_argument("--no-diff", action="store_true", help="do not print the diff")
    ap.add_argument("--pins", type=Path, default=PINS, help=argparse.SUPPRESS)  # tests only
    ap.add_argument("--repo", type=Path, default=REPO, help=argparse.SUPPRESS)  # tests only
    ap.add_argument("--skip-node4-guard", action="store_true", help=argparse.SUPPRESS)  # tests only
    a = ap.parse_args(argv)

    data = json.loads(a.pins.read_text())
    current_all = {rel: sha256_bytes((a.repo / rel).read_bytes()) for rel in data["sha256"]}
    stale = sorted(r for r, h in current_all.items() if h != data["sha256"][r])
    broken = log_problems(data)

    if not a.file:
        for rel in stale:
            print(f"STALE  {rel}  pinned {data['sha256'][rel][:12]}…  now {current_all[rel][:12]}…")
        for p in broken:
            print(f"LOG    {p}")
        if not stale and not broken:
            print(f"OK     all {len(current_all)} protected files match their pins")
        return 1 if stale or broken else 0

    errors = [f"--reason: {p}" for p in reason_problems(a.reason)]
    if broken:
        errors += [f"repin_log is already inconsistent, fix that first: {p}" for p in broken]
    files = sorted(set(a.file))
    for rel in files:
        if rel not in data["sha256"]:
            errors.append(f"{rel}: not a protected file (adding one is a reviewed edit of the map)")
        elif current_all[rel] == data["sha256"][rel]:
            errors.append(f"{rel}: still matches its pin; nothing to re-pin")
    if errors:
        for e in errors:
            print(f"refused: {e}", file=sys.stderr)
        return 2

    diff_bases = {}
    for rel in files:
        base = find_pinned_commit(rel, data["sha256"][rel], a.repo)
        diff_bases[rel] = base
        print(f"=== {rel}\n    pinned {data['sha256'][rel]}\n    now    {current_all[rel]}")
        if base is None:
            print("    WARNING: no commit in history has the pinned content; cannot show a diff")
        else:
            print(f"    pinned content last seen in {base[:12]}")
            if not a.no_diff:
                print(_git("diff", base, "--", rel, repo=a.repo).stdout or "    (no diff against working tree)")

    if NODE4 in files and not a.skip_node4_guard:
        print(f"=== {NODE4} is Node 4 scoring: running the golden and pinned-constants tests")
        if not node4_guard_passes():
            print("refused: Node 4 golden or pinned-constants test fails, so scoring or routing "
                  "changed. Review that change and update those tests deliberately first.",
                  file=sys.stderr)
            return 3

    for rel in sorted(set(stale) - set(files)):
        print(f"note: {rel} is also stale and is not being re-pinned")

    date = dt.datetime.now(dt.timezone.utc).date().isoformat()
    new = apply_repin(data, {r: current_all[r] for r in files}, a.reason, date, diff_bases)
    assert not log_problems(new), log_problems(new)
    if a.dry_run:
        print("dry run: protected_hashes.json not written")
        return 0
    tmp = a.pins.with_suffix(".json.tmp")
    tmp.write_text(dump(new))
    tmp.replace(a.pins)
    print(f"re-pinned {len(files)} file(s); commit protected_hashes.json together with the change")
    return 0


if __name__ == "__main__":
    sys.exit(main())
