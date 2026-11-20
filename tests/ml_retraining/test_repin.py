"""Tests for ``scripts/repin_protected_hashes.py`` and the committed ``repin_log``.

``test_committed_repin_log_is_consistent`` is the guard: a hand edit of the sha256 map in
``protected_hashes.json`` without a matching ``repin_log`` entry fails it. The rest test the
script on temporary copies; none of them touches the committed file or calls git.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PINS = Path(__file__).parent / "protected_hashes.json"
_spec = importlib.util.spec_from_file_location(
    "repin_protected_hashes", REPO / "scripts" / "repin_protected_hashes.py")
repin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(repin)

GOOD_REASON = "abc1234 added a display-only field after scoring; golden outputs unchanged"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---- the committed file ------------------------------------------------------------------------

def test_committed_repin_log_is_consistent():
    assert repin.log_problems(json.loads(PINS.read_text())) == []


def test_committed_file_is_in_canonical_format():
    text = PINS.read_text()
    assert repin.dump(json.loads(text)) == text


# ---- log_problems ------------------------------------------------------------------------------

@pytest.fixture
def data():
    return json.loads(PINS.read_text())


def test_hand_edit_of_map_without_log_is_caught(data):
    rel = sorted(data["sha256"])[0]
    data["sha256"][rel] = "0" * 64
    assert any("edited without the script" in p and rel in p for p in repin.log_problems(data))


def test_broken_chain_is_caught(data):
    rel = sorted(data["sha256"])[0]
    data["repin_log"].append({"kind": "repin", "file": rel, "old": "1" * 64, "new": "2" * 64,
                              "reason": GOOD_REASON, "date": "2026-01-01", "diff_base": None})
    data["sha256"][rel] = "2" * 64
    assert any("has old 111111111111" in p for p in repin.log_problems(data))


def test_logged_repin_with_weak_reason_is_caught(data):
    rel = sorted(data["sha256"])[0]
    old = data["sha256"][rel]
    data["repin_log"].append({"kind": "repin", "file": rel, "old": old, "new": "2" * 64,
                              "reason": "update hash", "date": "2026-01-01", "diff_base": None})
    data["sha256"][rel] = "2" * 64
    assert any("reason" in p for p in repin.log_problems(data))


def test_pinned_file_without_log_and_log_without_pin_are_caught(data):
    data["sha256"]["fraud_detect/new_file.py"] = "3" * 64
    data["repin_log"].append({"kind": "baseline", "file": "gone.py", "old": None, "new": "4" * 64,
                              "reason": "x", "date": "2026-01-01", "diff_base": None})
    probs = repin.log_problems(data)
    assert any("fraud_detect/new_file.py: pinned but has no repin_log entry" in p for p in probs)
    assert any("gone.py: in repin_log but not in the sha256 map" in p for p in probs)


def test_missing_log_is_caught(data):
    del data["repin_log"]
    assert repin.log_problems(data) == ["repin_log is missing or not a list"]


# ---- reason_problems ---------------------------------------------------------------------------

@pytest.mark.parametrize("reason", [None, "", "update hash", "Re-pin hashes after changes.",
                                    "fix tests", "refresh pinned hash for ci", "short but six words here ok"])
def test_weak_reasons_are_refused(reason):
    assert repin.reason_problems(reason)


def test_good_reason_is_accepted():
    assert repin.reason_problems(GOOD_REASON) == []


# ---- the CLI on a temporary repo copy ----------------------------------------------------------

@pytest.fixture
def sandbox(tmp_path):
    """Two protected files and a consistent pins file, outside the real repo."""
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    a, b = root / "pkg" / "a.py", root / "pkg" / "b.py"
    a.write_text("A = 1\n")
    b.write_text("B = 1\n")
    pins = {"note": "n", "node4_pinned": {}, "sha256": {"pkg/a.py": _sha(a.read_bytes()),
                                                        "pkg/b.py": _sha(b.read_bytes())}}
    pins["repin_log"] = [{"kind": "baseline", "file": f, "old": None, "new": h, "date": "2026-01-01",
                          "reason": "baseline", "diff_base": None} for f, h in sorted(pins["sha256"].items())]
    pf = tmp_path / "pins.json"
    pf.write_text(repin.dump(pins))
    return root, pf


def _run(root, pf, *args):
    return repin.main(["--pins", str(pf), "--repo", str(root), "--no-diff", "--skip-node4-guard", *args])


def test_status_ok_then_stale(sandbox, capsys):
    root, pf = sandbox
    assert _run(root, pf) == 0
    (root / "pkg" / "a.py").write_text("A = 2\n")
    assert _run(root, pf) == 1
    assert "STALE  pkg/a.py" in capsys.readouterr().out


def test_repin_records_reason_and_chains(sandbox):
    root, pf = sandbox
    (root / "pkg" / "a.py").write_text("A = 2\n")
    before = pf.read_text()
    assert _run(root, pf, "--file", "pkg/a.py", "--reason", GOOD_REASON) == 0
    after = json.loads(pf.read_text())
    assert after["sha256"]["pkg/a.py"] == _sha(b"A = 2\n")
    last = after["repin_log"][-1]
    assert (last["kind"], last["file"], last["reason"]) == ("repin", "pkg/a.py", GOOD_REASON)
    assert last["old"] == json.loads(before)["sha256"]["pkg/a.py"]
    assert repin.log_problems(after) == []
    assert _run(root, pf) == 0


def test_dry_run_writes_nothing(sandbox):
    root, pf = sandbox
    (root / "pkg" / "a.py").write_text("A = 2\n")
    before = pf.read_text()
    assert _run(root, pf, "--file", "pkg/a.py", "--reason", GOOD_REASON, "--dry-run") == 0
    assert pf.read_text() == before


@pytest.mark.parametrize("args, message", [
    (["--file", "pkg/a.py"], "--reason"),
    (["--file", "pkg/a.py", "--reason", "update hash"], "--reason"),
    (["--file", "pkg/b.py", "--reason", GOOD_REASON], "still matches its pin"),
    (["--file", "pkg/new.py", "--reason", GOOD_REASON], "not a protected file"),
])
def test_refusals_write_nothing(sandbox, capsys, args, message):
    root, pf = sandbox
    (root / "pkg" / "a.py").write_text("A = 2\n")
    (root / "pkg" / "new.py").write_text("")
    before = pf.read_text()
    assert _run(root, pf, *args) == 2
    assert message in capsys.readouterr().err
    assert pf.read_text() == before


def test_refuses_when_log_already_broken(sandbox, capsys):
    root, pf = sandbox
    data = json.loads(pf.read_text())
    data["sha256"]["pkg/b.py"] = "9" * 64  # hand edit
    pf.write_text(repin.dump(data))
    (root / "pkg" / "a.py").write_text("A = 2\n")
    assert _run(root, pf, "--file", "pkg/a.py", "--reason", GOOD_REASON) == 2
    assert "already inconsistent" in capsys.readouterr().err


def test_only_named_files_are_repinned(sandbox, capsys):
    root, pf = sandbox
    (root / "pkg" / "a.py").write_text("A = 2\n")
    (root / "pkg" / "b.py").write_text("B = 2\n")
    assert _run(root, pf, "--file", "pkg/a.py", "--reason", GOOD_REASON) == 0
    after = json.loads(pf.read_text())
    assert after["sha256"]["pkg/b.py"] == _sha(b"B = 1\n")
    assert "pkg/b.py is also stale" in capsys.readouterr().out


def test_node4_guard_blocks_repin(sandbox, monkeypatch, capsys):
    root, pf = sandbox
    node4 = root / repin.NODE4
    node4.parent.mkdir(parents=True)
    node4.write_text("X = 1\n")
    data = json.loads(pf.read_text())
    data["sha256"][repin.NODE4] = _sha(b"X = 1\n")
    data["repin_log"].append({"kind": "baseline", "file": repin.NODE4, "old": None,
                              "new": _sha(b"X = 1\n"), "date": "2026-01-01", "reason": "b",
                              "diff_base": None})
    pf.write_text(repin.dump(data))
    node4.write_text("X = 2\n")
    monkeypatch.setattr(repin, "node4_guard_passes", lambda: False)
    before = pf.read_text()
    rc = repin.main(["--pins", str(pf), "--repo", str(root), "--no-diff",
                     "--file", repin.NODE4, "--reason", GOOD_REASON])
    assert rc == 3
    assert "golden or pinned-constants test fails" in capsys.readouterr().err
    assert pf.read_text() == before
