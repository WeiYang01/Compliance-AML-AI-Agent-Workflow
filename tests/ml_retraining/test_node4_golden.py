"""Golden-output test for Node 4 ``synthesize()``.

``protected_hashes.json`` pins the bytes of ``risk_synthesis.py`` and
``test_node4_constants_and_literals_equal_pinned`` pins its numeric literals.
Neither catches a logic change that keeps the same numbers (a flipped ``<`` /
``<=``, a reordered branch in ``_route``, a changed condition). This test does:
it feeds fixed, hand-built inputs through ``synthesize()`` and pins the exact
risk, confidence, bands, routing, routing reason and summary.

Each case targets one branch or boundary of ``_compute_risk``,
``_compute_confidence`` or ``_route``; the comment on each case gives the
arithmetic, so a reviewer can check the pinned values without running code.

If this test fails, Node 4 scoring or routing changed. That is a behaviour
change needing review, not a test to update. Change EXPECTED only in the same
commit as a deliberate, reviewed Node 4 change, and say why in the message.
"""
from __future__ import annotations

from datetime import date

import pytest

from fraud_detect.models import (
    AgentReport,
    DocumentMetadata,
    Finding,
    IngestionResult,
    Severity,
    Transaction,
)
from fraud_detect.nodes.risk_synthesis import _band_conf, _band_risk, synthesize

N1, N2, N3 = "Node1:Ingestion", "Node2:VisualForensics", "Node3:FinancialLogic"
L, M, H = Severity.LOW, Severity.MEDIUM, Severity.HIGH


def _f(agent: str, sev: Severity, weight: float, code: str = "X") -> Finding:
    return Finding(agent=agent, code=code, title=code, severity=sev, weight=weight,
                   evidence="golden")


def _ing(name: str, *, txns: bool = True, dq: tuple[Finding, ...] = ()) -> IngestionResult:
    rows = [Transaction(index=0, date=date(2026, 1, 5), amount=1000.0, balance=1000.0)] if txns else []
    return IngestionResult(
        source_path=f"golden/{name}.pdf", page_count=1, raw_text="", metadata=DocumentMetadata(),
        account_holder="Golden Test Pte Ltd", statement_period_start=date(2026, 1, 1),
        statement_period_end=date(2026, 1, 31), transactions=rows,
        data_quality_findings=list(dq),
    )


def _rep(*findings: Finding) -> list[AgentReport]:
    by_agent: dict[str, list[Finding]] = {}
    for f in findings:
        by_agent.setdefault(f.agent, []).append(f)
    return [AgentReport(agent=a, findings=fs) for a, fs in by_agent.items()]


#: name -> (IngestionResult, agent reports)
CASES = {
    # No findings: risk 1; conf 85. AUTO_CLEAR.
    "clean": (_ing("clean"), []),
    # No findings, no rows: conf 85-25 = 60, exactly MIN_CONFIDENCE_FOR_AUTO (not < 60). AUTO_CLEAR.
    "no_rows_conf_at_threshold": (_ing("no_rows", txns=False), []),
    # One LOW DQ finding (w3), no rows: conf 85-8-25 = 52 < 60. Risk 3. HUMAN_REVIEW on confidence.
    "low_conf_escalates": (_ing("low_conf", txns=False, dq=(_f(N1, L, 3, "DQ_A"),)), []),
    # One LOW Node3 finding w25: risk 25 = AUTO_CLEAR_MAX_RISK; conf 85-12 (one dimension)
    # -10 (thin evidence, 20<=25<=40) = 63. AUTO_CLEAR at the risk boundary.
    "risk_at_auto_clear_max": (_ing("r25"), _rep(_f(N3, L, 25))),
    # Two LOW Node3 findings w13: risk 26 > 25, no MEDIUM+; conf 85-12 = 73. HUMAN_REVIEW on risk.
    "risk_just_over_auto_clear": (_ing("r26"), _rep(_f(N3, L, 13), _f(N3, L, 13, "Y"))),
    # One MEDIUM Node2 finding w5: risk 5 floored to 26 (material); conf 85-12 (one dimension)
    # -10 (thin evidence: the floored 26 is in 20..40) = 63. HUMAN_REVIEW on material, band MEDIUM.
    "medium_floor_and_material": (_ing("med"), _rep(_f(N2, M, 5))),
    # HIGH from Node2 (w40) and Node3 (w45): 85 + 10 corroboration = 95; conf 85+10 = 95.
    "two_agents_high_corroborated": (_ing("hh"), _rep(_f(N2, H, 40), _f(N3, H, 45))),
    # Weights 80 + 80 + 10 bonus = 170, clamped to 100.
    "risk_clamped_to_100": (_ing("clamp"), _rep(_f(N2, H, 80), _f(N3, H, 80))),
    # Six LOW DQ findings w1: DQ penalty min(40, 48) = 40; risk 6; conf 85-40 = 45 < 60.
    "dq_penalty_capped_at_40": (_ing("dq6", dq=tuple(_f(N1, L, 1, f"DQ_{i}") for i in range(6))), []),
    # LOW Node3 w12.6: rounds to 13, not truncated to 12; conf 85-12 = 73. AUTO_CLEAR.
    "fraction_rounds_not_truncates": (_ing("frac"), _rep(_f(N3, L, 12.6))),
    # LOW Node3 w12.5: round(12.5) = 12 (banker's rounding); conf 85-12 = 73. AUTO_CLEAR.
    "half_weight_rounds_to_even": (_ing("half"), _rep(_f(N3, L, 12.5))),
    # One HIGH Node2 w50: risk 50 (MEDIUM band top edge); conf 85-12 = 73.
    "risk_50_band_edge": (_ing("r50"), _rep(_f(N2, H, 50))),
    # HIGH Node2 w76 alone: risk 76, VERY_HIGH; conf 85-12 = 73.
    "risk_76_very_high": (_ing("r76"), _rep(_f(N2, H, 76))),
    # Thin-evidence window (exactly one finding, 20 <= risk <= 40), both sides of both edges.
    # One LOW Node3 finding each; conf 85-12 = 73 outside the window, 73-10 = 63 inside.
    "thin_window_below_19": (_ing("t19"), _rep(_f(N3, L, 19))),
    "thin_window_low_edge_20": (_ing("t20"), _rep(_f(N3, L, 20))),
    "thin_window_high_edge_40": (_ing("t40"), _rep(_f(N3, L, 40))),
    "thin_window_above_41": (_ing("t41"), _rep(_f(N3, L, 41))),
    # One INFO Node2 finding w0: round(0) = 0, floored to 1; conf 85-12 = 73. AUTO_CLEAR.
    "zero_weight_risk_floored_to_1": (_ing("zero"), _rep(_f(N2, Severity.INFO, 0))),
    # DQ LOW w2 (Node1) + LOW Node3 w20: risk 22; Node1 and Node3 corroborate (+10);
    # DQ -8; only financial flagged among visual/financial (-12): 85-8+10-12 = 75. AUTO_CLEAR.
    "dq_plus_financial_corroboration": (_ing("dqfin", dq=(_f(N1, L, 2, "DQ_A"),)), _rep(_f(N3, L, 20))),
}

EXPECTED: dict = {
    'clean': {
        'case_id': 'CASE-B539DBFF48',
        'confidence_band': 'VERY_HIGH',
        'confidence_score': 85,
        'risk_band': 'LOW',
        'risk_score': 1,
        'routing': 'AUTO_CLEAR',
        'routing_reason': 'Low risk (1) with sufficient confidence (85) and no material findings; auto-generating an approval recommendation.',
        'summary': '0 finding(s) across the visual-forensics and financial-logic agents (0 high-severity). Risk 1/100 (LOW); confidence 85/100 (VERY_HIGH). Confidence rationale: no adjustments.',
    },
    'dq_penalty_capped_at_40': {
        'case_id': 'CASE-3754BF13A6',
        'confidence_band': 'MEDIUM',
        'confidence_score': 45,
        'risk_band': 'LOW',
        'risk_score': 6,
        'routing': 'HUMAN_REVIEW',
        'routing_reason': 'Confidence (45) below the auto-clear threshold (60); escalating to a human reviewer.',
        'summary': '6 finding(s) across the visual-forensics and financial-logic agents (0 high-severity). Risk 6/100 (LOW); confidence 45/100 (MEDIUM). Confidence rationale: -40: 6 data-quality issue(s) reduce certainty.',
    },
    'dq_plus_financial_corroboration': {
        'case_id': 'CASE-FB09D43C98',
        'confidence_band': 'HIGH',
        'confidence_score': 75,
        'risk_band': 'LOW',
        'risk_score': 22,
        'routing': 'AUTO_CLEAR',
        'routing_reason': 'Low risk (22) with sufficient confidence (75) and no material findings; auto-generating an approval recommendation.',
        'summary': '2 finding(s) across the visual-forensics and financial-logic agents (0 high-severity). Risk 22/100 (LOW); confidence 75/100 (HIGH). Confidence rationale: -8: 1 data-quality issue(s) reduce certainty; +10: independent agents corroborate the finding; -12: evidence limited to a single dimension (visual OR financial).',
    },
    'fraction_rounds_not_truncates': {
        'case_id': 'CASE-BE754D14C7',
        'confidence_band': 'HIGH',
        'confidence_score': 73,
        'risk_band': 'LOW',
        'risk_score': 13,
        'routing': 'AUTO_CLEAR',
        'routing_reason': 'Low risk (13) with sufficient confidence (73) and no material findings; auto-generating an approval recommendation.',
        'summary': '1 finding(s) across the visual-forensics and financial-logic agents (0 high-severity). Risk 13/100 (LOW); confidence 73/100 (HIGH). Confidence rationale: -12: evidence limited to a single dimension (visual OR financial).',
    },
    'half_weight_rounds_to_even': {
        'case_id': 'CASE-0E8EF479ED',
        'confidence_band': 'HIGH',
        'confidence_score': 73,
        'risk_band': 'LOW',
        'risk_score': 12,
        'routing': 'AUTO_CLEAR',
        'routing_reason': 'Low risk (12) with sufficient confidence (73) and no material findings; auto-generating an approval recommendation.',
        'summary': '1 finding(s) across the visual-forensics and financial-logic agents (0 high-severity). Risk 12/100 (LOW); confidence 73/100 (HIGH). Confidence rationale: -12: evidence limited to a single dimension (visual OR financial).',
    },
    'low_conf_escalates': {
        'case_id': 'CASE-8BE5E3C2B8',
        'confidence_band': 'HIGH',
        'confidence_score': 52,
        'risk_band': 'LOW',
        'risk_score': 3,
        'routing': 'HUMAN_REVIEW',
        'routing_reason': 'Confidence (52) below the auto-clear threshold (60); escalating to a human reviewer.',
        'summary': '1 finding(s) across the visual-forensics and financial-logic agents (0 high-severity). Risk 3/100 (LOW); confidence 52/100 (HIGH). Confidence rationale: -8: 1 data-quality issue(s) reduce certainty; -25: no transactions parsed.',
    },
    'medium_floor_and_material': {
        'case_id': 'CASE-350BC5DEF8',
        'confidence_band': 'HIGH',
        'confidence_score': 63,
        'risk_band': 'MEDIUM',
        'risk_score': 26,
        'routing': 'HUMAN_REVIEW',
        'routing_reason': '1 inconsistency/tampering finding(s) of MEDIUM+ severity present; escalating to a human reviewer even though the numeric risk is 26.',
        'summary': '1 finding(s) across the visual-forensics and financial-logic agents (0 high-severity). Risk 26/100 (MEDIUM); confidence 63/100 (HIGH). Confidence rationale: -12: evidence limited to a single dimension (visual OR financial); -10: thin evidence near a risk-band boundary.',
    },
    'no_rows_conf_at_threshold': {
        'case_id': 'CASE-980BB44196',
        'confidence_band': 'HIGH',
        'confidence_score': 60,
        'risk_band': 'LOW',
        'risk_score': 1,
        'routing': 'AUTO_CLEAR',
        'routing_reason': 'Low risk (1) with sufficient confidence (60) and no material findings; auto-generating an approval recommendation.',
        'summary': '0 finding(s) across the visual-forensics and financial-logic agents (0 high-severity). Risk 1/100 (LOW); confidence 60/100 (HIGH). Confidence rationale: -25: no transactions parsed.',
    },
    'risk_50_band_edge': {
        'case_id': 'CASE-D9C3193748',
        'confidence_band': 'HIGH',
        'confidence_score': 73,
        'risk_band': 'MEDIUM',
        'risk_score': 50,
        'routing': 'HUMAN_REVIEW',
        'routing_reason': '1 inconsistency/tampering finding(s) of MEDIUM+ severity present; escalating to a human reviewer even though the numeric risk is 50.',
        'summary': '1 finding(s) across the visual-forensics and financial-logic agents (1 high-severity). Risk 50/100 (MEDIUM); confidence 73/100 (HIGH). Confidence rationale: -12: evidence limited to a single dimension (visual OR financial).',
    },
    'risk_76_very_high': {
        'case_id': 'CASE-2872EA92DB',
        'confidence_band': 'HIGH',
        'confidence_score': 73,
        'risk_band': 'VERY_HIGH',
        'risk_score': 76,
        'routing': 'HUMAN_REVIEW',
        'routing_reason': '1 inconsistency/tampering finding(s) of MEDIUM+ severity present; escalating to a human reviewer even though the numeric risk is 76.',
        'summary': '1 finding(s) across the visual-forensics and financial-logic agents (1 high-severity). Risk 76/100 (VERY_HIGH); confidence 73/100 (HIGH). Confidence rationale: -12: evidence limited to a single dimension (visual OR financial).',
    },
    'risk_at_auto_clear_max': {
        'case_id': 'CASE-DF813C81C2',
        'confidence_band': 'HIGH',
        'confidence_score': 63,
        'risk_band': 'LOW',
        'risk_score': 25,
        'routing': 'AUTO_CLEAR',
        'routing_reason': 'Low risk (25) with sufficient confidence (63) and no material findings; auto-generating an approval recommendation.',
        'summary': '1 finding(s) across the visual-forensics and financial-logic agents (0 high-severity). Risk 25/100 (LOW); confidence 63/100 (HIGH). Confidence rationale: -12: evidence limited to a single dimension (visual OR financial); -10: thin evidence near a risk-band boundary.',
    },
    'risk_clamped_to_100': {
        'case_id': 'CASE-5D2A6FF64F',
        'confidence_band': 'VERY_HIGH',
        'confidence_score': 95,
        'risk_band': 'VERY_HIGH',
        'risk_score': 100,
        'routing': 'HUMAN_REVIEW',
        'routing_reason': '2 inconsistency/tampering finding(s) of MEDIUM+ severity present; escalating to a human reviewer even though the numeric risk is 100.',
        'summary': '2 finding(s) across the visual-forensics and financial-logic agents (2 high-severity). Risk 100/100 (VERY_HIGH); confidence 95/100 (VERY_HIGH). Confidence rationale: +10: independent agents corroborate the finding.',
    },
    'risk_just_over_auto_clear': {
        'case_id': 'CASE-19D15AB8BB',
        'confidence_band': 'HIGH',
        'confidence_score': 73,
        'risk_band': 'MEDIUM',
        'risk_score': 26,
        'routing': 'HUMAN_REVIEW',
        'routing_reason': 'Risk score (26) exceeds the auto-clear threshold (25); escalating to a human reviewer.',
        'summary': '2 finding(s) across the visual-forensics and financial-logic agents (0 high-severity). Risk 26/100 (MEDIUM); confidence 73/100 (HIGH). Confidence rationale: -12: evidence limited to a single dimension (visual OR financial).',
    },
    'thin_window_above_41': {
        'case_id': 'CASE-6436135E7F',
        'confidence_band': 'HIGH',
        'confidence_score': 73,
        'risk_band': 'MEDIUM',
        'risk_score': 41,
        'routing': 'HUMAN_REVIEW',
        'routing_reason': 'Risk score (41) exceeds the auto-clear threshold (25); escalating to a human reviewer.',
        'summary': '1 finding(s) across the visual-forensics and financial-logic agents (0 high-severity). Risk 41/100 (MEDIUM); confidence 73/100 (HIGH). Confidence rationale: -12: evidence limited to a single dimension (visual OR financial).',
    },
    'thin_window_below_19': {
        'case_id': 'CASE-963EDE14FB',
        'confidence_band': 'HIGH',
        'confidence_score': 73,
        'risk_band': 'LOW',
        'risk_score': 19,
        'routing': 'AUTO_CLEAR',
        'routing_reason': 'Low risk (19) with sufficient confidence (73) and no material findings; auto-generating an approval recommendation.',
        'summary': '1 finding(s) across the visual-forensics and financial-logic agents (0 high-severity). Risk 19/100 (LOW); confidence 73/100 (HIGH). Confidence rationale: -12: evidence limited to a single dimension (visual OR financial).',
    },
    'thin_window_high_edge_40': {
        'case_id': 'CASE-6B7FDBDE32',
        'confidence_band': 'HIGH',
        'confidence_score': 63,
        'risk_band': 'MEDIUM',
        'risk_score': 40,
        'routing': 'HUMAN_REVIEW',
        'routing_reason': 'Risk score (40) exceeds the auto-clear threshold (25); escalating to a human reviewer.',
        'summary': '1 finding(s) across the visual-forensics and financial-logic agents (0 high-severity). Risk 40/100 (MEDIUM); confidence 63/100 (HIGH). Confidence rationale: -12: evidence limited to a single dimension (visual OR financial); -10: thin evidence near a risk-band boundary.',
    },
    'thin_window_low_edge_20': {
        'case_id': 'CASE-CB0F48DAA0',
        'confidence_band': 'HIGH',
        'confidence_score': 63,
        'risk_band': 'LOW',
        'risk_score': 20,
        'routing': 'AUTO_CLEAR',
        'routing_reason': 'Low risk (20) with sufficient confidence (63) and no material findings; auto-generating an approval recommendation.',
        'summary': '1 finding(s) across the visual-forensics and financial-logic agents (0 high-severity). Risk 20/100 (LOW); confidence 63/100 (HIGH). Confidence rationale: -12: evidence limited to a single dimension (visual OR financial); -10: thin evidence near a risk-band boundary.',
    },
    'two_agents_high_corroborated': {
        'case_id': 'CASE-008C55828C',
        'confidence_band': 'VERY_HIGH',
        'confidence_score': 95,
        'risk_band': 'VERY_HIGH',
        'risk_score': 95,
        'routing': 'HUMAN_REVIEW',
        'routing_reason': '2 inconsistency/tampering finding(s) of MEDIUM+ severity present; escalating to a human reviewer even though the numeric risk is 95.',
        'summary': '2 finding(s) across the visual-forensics and financial-logic agents (2 high-severity). Risk 95/100 (VERY_HIGH); confidence 95/100 (VERY_HIGH). Confidence rationale: +10: independent agents corroborate the finding.',
    },
    'zero_weight_risk_floored_to_1': {
        'case_id': 'CASE-F8CF32003C',
        'confidence_band': 'HIGH',
        'confidence_score': 73,
        'risk_band': 'LOW',
        'risk_score': 1,
        'routing': 'AUTO_CLEAR',
        'routing_reason': 'Low risk (1) with sufficient confidence (73) and no material findings; auto-generating an approval recommendation.',
        'summary': '1 finding(s) across the visual-forensics and financial-logic agents (0 high-severity). Risk 1/100 (LOW); confidence 73/100 (HIGH). Confidence rationale: -12: evidence limited to a single dimension (visual OR financial).',
    },
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_synthesize_golden(name):
    ing, reports = CASES[name]
    r = synthesize(ing, reports)
    got = {
        "case_id": r.case_id,
        "risk_score": r.risk_score,
        "risk_band": r.risk_band.value,
        "confidence_score": r.confidence_score,
        "confidence_band": r.confidence_band.value,
        "routing": r.routing.value,
        "routing_reason": r.routing_reason,
        "summary": r.summary,
    }
    assert got == EXPECTED[name]


def test_every_case_is_pinned():
    assert set(CASES) == set(EXPECTED)


#: Band edges, both sides: 1..25 LOW, 26..50 MEDIUM, 51..75 HIGH, 76..100 VERY_HIGH.
BAND_EDGES = {1: "LOW", 25: "LOW", 26: "MEDIUM", 50: "MEDIUM", 51: "HIGH", 75: "HIGH",
              76: "VERY_HIGH", 100: "VERY_HIGH"}


@pytest.mark.parametrize("score", sorted(BAND_EDGES))
def test_band_edges(score):
    assert _band_risk(score).value == BAND_EDGES[score]
    assert _band_conf(score).value == BAND_EDGES[score]
