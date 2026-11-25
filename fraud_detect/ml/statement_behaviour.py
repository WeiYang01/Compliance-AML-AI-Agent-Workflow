"""Statement-level behaviour features for Node 6a.

Node 6a's existing model scores one transaction at a time, so it cannot see patterns that
only exist across a whole statement. These features describe the statement as a whole,
from the transactions Node 1 parsed. They target content fabrications that leave no trace
in the file: inflated turnover, copied receipts, invented payers.

Where a rule-based check looks at a related idea, the feature is deliberately different:
Node 3's turnover-cost rule compares fixed ratios, its duplicate rule matches invoice
references, its sudden-deposit rule looks at single credits. These features give the
model shape and proportion instead, and leave the yes/no decisions to the rules.
"""
from __future__ import annotations

import math
import re
import statistics
from collections import Counter, defaultdict
from typing import Optional

FEATURE_NAMES = [
    "month_credit_peak_ratio",       # busiest month's receipts vs a typical month
    "month_credit_spread",           # how uneven monthly receipts are (coefficient of variation)
    "credit_debit_ratio",            # receipts vs spending over the statement
    "month_credit_debit_peak",       # the month where receipts outrun spending the most
    "repeat_across_months_share",    # receipts whose description AND amount recur in another month
    "repeat_amount_share",           # receipts whose exact amount appears more than once
    "benford_deviation",             # first digits of receipts vs Benford's law
    "round_amount_share",            # receipts that are whole hundreds
    "top_payer_share",               # share of receipt value from the largest payer
    "one_month_payer_share",         # share of receipt value from payers seen in only one month
    "credits_per_month_peak_ratio",  # busiest month's count of receipts vs a typical month
]

_BENFORD = [math.log10(1 + 1 / d) for d in range(1, 10)]
_REF = re.compile(r"\b(?:INV|REF|NO)[-\s#:]*[\w/-]+", re.IGNORECASE)
_DIGITS = re.compile(r"\d+")


def _payer(description: Optional[str]) -> str:
    text = _REF.sub(" ", description or "")
    text = _DIGITS.sub(" ", text)
    return " ".join(text.upper().split())


def _ratio(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    typical = statistics.median(values)
    return max(values) / typical if typical > 0 else 1.0


def extract(transactions) -> Optional[dict[str, float]]:
    """Behaviour features of one statement's parsed transactions, or ``None``."""
    rows = [t for t in transactions if t.amount is not None and t.date is not None]
    credits = [t for t in rows if t.amount > 0]
    if len(rows) < 5 or not credits:
        return None
    month = lambda t: (t.date.year, t.date.month)  # noqa: E731
    month_credit: dict = defaultdict(float)
    month_debit: dict = defaultdict(float)
    month_count: Counter = Counter()
    for t in rows:
        if t.amount > 0:
            month_credit[month(t)] += t.amount
            month_count[month(t)] += 1
        else:
            month_debit[month(t)] += -t.amount
    months = sorted(set(month_credit) | set(month_debit))
    credit_values = [month_credit[m] for m in months]
    mean_credit = statistics.fmean(credit_values)

    pair_months: dict = defaultdict(set)
    for t in credits:
        pair_months[(_payer(t.description), round(t.amount, 2))].add(month(t))
    amounts = Counter(round(t.amount, 2) for t in credits)

    first = Counter(int(str(int(t.amount * 100)).lstrip("0")[0]) for t in credits
                    if t.amount >= 1)
    n_first = sum(first.values())
    benford = (statistics.fmean(abs(first[d] / n_first - _BENFORD[d - 1]) for d in range(1, 10))
               if n_first >= 20 else math.nan)

    payer_value: Counter = Counter()
    payer_months: dict = defaultdict(set)
    for t in credits:
        payer_value[_payer(t.description)] += t.amount
        payer_months[_payer(t.description)].add(month(t))
    total_credit = sum(payer_value.values())
    total_debit = sum(month_debit.values())

    month_ratios = [month_credit[m] / month_debit[m] for m in months if month_debit[m] > 0]
    return {
        "month_credit_peak_ratio": _ratio(credit_values),
        "month_credit_spread": (statistics.pstdev(credit_values) / mean_credit
                                if mean_credit > 0 else 0.0),
        "credit_debit_ratio": total_credit / total_debit if total_debit > 0 else math.nan,
        "month_credit_debit_peak": (max(month_ratios) / statistics.median(month_ratios)
                                    if len(month_ratios) >= 2 else math.nan),
        "repeat_across_months_share": sum(1 for t in credits
                                          if len(pair_months[(_payer(t.description),
                                                              round(t.amount, 2))]) > 1)
        / len(credits),
        "repeat_amount_share": sum(1 for t in credits if amounts[round(t.amount, 2)] > 1)
        / len(credits),
        "benford_deviation": benford,
        "round_amount_share": sum(1 for t in credits if t.amount % 100 == 0) / len(credits),
        "top_payer_share": max(payer_value.values()) / total_credit if total_credit else 0.0,
        "one_month_payer_share": (sum(v for p, v in payer_value.items() if len(payer_months[p]) == 1)
                                  / total_credit if total_credit and len(months) > 1 else 0.0),
        "credits_per_month_peak_ratio": _ratio([float(month_count[m]) for m in months]),
    }


# ---------------------------------------------------------------------------
# The Stage B feature set for Node 6a
# ---------------------------------------------------------------------------
#: The statement features Stage B trains Node 6a on. They need only dates and signed
#: amounts, which both the corpus's rows.csv and a parsed PDF carry, so training and the
#: live node see the same inputs. Left out on purpose:
#:   * the payer features (repeat_across_months_share, top_payer_share,
#:     one_month_payer_share): rows.csv has no descriptions;
#:   * round_amount_share: the Feature_Audit lists round amounts as a generator tell
#:     (generated amounts carry cents, so round ones mark inserted rows).
STAGE_B_FEATURES = [
    "month_credit_peak_ratio",
    "month_credit_spread",
    "credit_debit_ratio",
    "month_credit_debit_peak",
    "repeat_amount_share",
    "benford_deviation",
    "credits_per_month_peak_ratio",
]
#: Value for a feature that cannot be computed (no debits, fewer than 20 receipts for
#: Benford, one month only). Every real value is non-negative, so the model can tell it apart.
NOT_COMPUTABLE = -1.0


def stage_b_features(records) -> Optional[dict[str, float]]:
    """The Stage B features from ``(date, signed amount)`` pairs, or ``None`` if too few.

    Used both by the retraining pipeline (rows.csv) and by the live node (parsed
    transactions), so the two always compute the same values.
    """
    from types import SimpleNamespace

    rows = [SimpleNamespace(date=d, amount=float(a), description=None)
            for d, a in records if d is not None and a is not None]
    full = extract(rows)
    if full is None:
        return None
    return {k: (NOT_COMPUTABLE if math.isnan(full[k]) else float(full[k])) for k in STAGE_B_FEATURES}
