"""The sample-quantile (SQ) test for a single firm and a single event.

Gelbach, Helland and Klick (2013, American Law and Economics Review 15(2), 495-541) propose testing a
single event-day abnormal return against the sample quantiles of the pre-event fitted excess returns
(the estimation-window residuals) instead of against a normal distribution. The test does not assume a
shape for the residual distribution, which matters for single-firm studies: daily residuals are
fat-tailed, so normal critical values are wrong in the tails.

For a one-day window this module compares the event-day abnormal return with the estimation-window
residuals. For a window of L days it compares the cumulative abnormal return with the sums of L
consecutive estimation-window residuals (overlapping, so there are n - L + 1 of them). The multi-day
version is this project's extension of the one-day test; it is not taken from the paper.

p-values count the observed value as one more draw from the reference distribution (the usual rank
convention, which keeps the test from over-rejecting in finite samples when the event return and the
residuals are exchangeable):

    p_lower = (1 + #{s <= x}) / (N + 1),   p_upper = (1 + #{s >= x}) / (N + 1),
    p_two_sided = min(1, 2 min(p_lower, p_upper)).

With N reference values the smallest attainable one-sided p-value is 1/(N + 1): an abnormal return beyond
every estimation-window residual gets exactly that, so with 250 residuals no one-day result can be
"more significant" than p = 0.004 one-sided. For L > 1 the overlapping window sums are not independent,
so the multi-day p-values are approximate. `reject` applies the quantile form of the test: reject at
level alpha (two-sided) if x is below the alpha/2 sample quantile or above the 1 - alpha/2 sample quantile.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def window_sums(residuals: np.ndarray, length: int) -> np.ndarray:
    """Sums of `length` consecutive residuals (overlapping windows): n - length + 1 values."""
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    if length < 1:
        raise ValueError("length must be at least 1")
    if len(r) < length:
        raise ValueError(f"need at least {length} residuals, got {len(r)}")
    if length == 1:
        return r.copy()
    c = np.concatenate([[0.0], np.cumsum(r)])
    return c[length:] - c[:-length]


@dataclass(frozen=True)
class SQResult:
    """One SQ test: the observed value, the p-values and the size of the reference distribution."""

    value: float
    p_lower: float
    p_upper: float
    p_two_sided: float
    reference_size: int
    window_length: int

    @property
    def min_p(self) -> float:
        """The smallest one-sided p-value the reference distribution can give: 1/(N + 1)."""
        return 1.0 / (self.reference_size + 1)


def sq_test(value: float, residuals: np.ndarray, length: int = 1) -> SQResult:
    """SQ test of `value` (an abnormal return, or a cumulative one over `length` days) against the
    estimation-window `residuals`."""
    reference = window_sums(residuals, length)
    n = len(reference)
    if not np.isfinite(value):
        return SQResult(float("nan"), float("nan"), float("nan"), float("nan"), n, length)
    p_lower = float((1 + np.sum(reference <= value)) / (n + 1))
    p_upper = float((1 + np.sum(reference >= value)) / (n + 1))
    p_two = min(1.0, 2.0 * min(p_lower, p_upper))
    return SQResult(float(value), p_lower, p_upper, p_two, n, length)


def reject(value: float, residuals: np.ndarray, alpha: float = 0.05, length: int = 1) -> bool:
    """Quantile form of the two-sided SQ test at level `alpha`."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    reference = window_sums(residuals, length)
    lo, hi = np.quantile(reference, [alpha / 2, 1 - alpha / 2])
    return bool(value < lo or value > hi)
