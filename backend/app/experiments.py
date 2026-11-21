"""Online A/B testing: stable assignment of users to groups, and the statistics to compare them."""

from __future__ import annotations

import hashlib
import math
import uuid

from app.settings import Settings


def assign(user_id: uuid.UUID, settings: Settings) -> str:
    """The user's group in the current experiment: a stable hash of (experiment, user), so a user stays
    in the same group on every device and request, and groups are balanced in expectation."""
    groups = list(settings.ab_variants)
    if not groups:
        return "control"
    if not settings.ab_experiment:
        return groups[0]
    digest = hashlib.sha256(f"{settings.ab_experiment}:{user_id}".encode()).digest()
    return groups[int.from_bytes(digest[:8], "big") % len(groups)]


def ctr_test(clicks_a: int, shown_a: int, clicks_b: int, shown_b: int) -> dict:
    """Two-proportion z-test on click-through rates (B vs A), with a 95% confidence interval for the
    difference. Returns None fields when a group has no impressions yet."""
    if not shown_a or not shown_b:
        return {"difference": None, "ci95": None, "z": None, "p_value": None}
    pa, pb = clicks_a / shown_a, clicks_b / shown_b
    pooled = (clicks_a + clicks_b) / (shown_a + shown_b)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / shown_a + 1 / shown_b))
    z = (pb - pa) / se_pooled if se_pooled > 0 else 0.0
    se = math.sqrt(pa * (1 - pa) / shown_a + pb * (1 - pb) / shown_b)
    return {
        "difference": pb - pa,
        "ci95": [pb - pa - 1.96 * se, pb - pa + 1.96 * se],
        "z": z,
        "p_value": math.erfc(abs(z) / math.sqrt(2)),
    }
