"""Typographic features for Node 6b, from the Node 2 typography measurements.

The measurements live in ``fraud_detect.nodes.typography_forensics``, which also turns
them into the Node 2 rules ``VF_SPACING_OUTLIER`` and ``VF_COLUMN_MISALIGNED``. Sharing
one implementation keeps the rule and the model features identical. On synthetic data
the rule does better than a model on these features (see FRAUD_DETECT_README.md,
"Step 3"); the features stay available for a model trained on real statements.
"""
from __future__ import annotations

from typing import Optional

from fraud_detect.nodes.typography_forensics import SPACING_OUTLIER, measure

FEATURE_NAMES = [
    "spacing_irregularity_max",    # worst number's spacing, above the document's typical spacing
    "spacing_irregular_tokens",    # numbers whose spacing stands out from the document
    "column_misaligned_tokens",    # money values ending just off their column's right edge
    "column_misalignment_max",     # worst such offset, in font sizes
]


def extract(pdf_bytes: bytes) -> Optional[dict[str, float]]:
    """Typographic features of one PDF, or ``None`` if it cannot be read."""
    m = measure(pdf_bytes)
    if m is None:
        return None
    excess = [t.value for t in m.spacing_excess()]
    return {
        "spacing_irregularity_max": max([0.0] + excess),
        "spacing_irregular_tokens": float(sum(1 for e in excess if e > SPACING_OUTLIER)),
        "column_misaligned_tokens": float(len(m.misaligned)),
        "column_misalignment_max": max((t.value for t in m.misaligned), default=0.0),
    }
