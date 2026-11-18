"""Tests of the mean CAR across events.

- Cross-sectional t: mean CAR over its cross-sectional standard error, t with N - 1 degrees of freedom.
- BMP (Boehmer, Musumeci and Poulsen 1991, Journal of Financial Economics 30(2), 253-272): the same
  t-test applied to standardised CARs, SCAR_i = CAR_i / se_i, where se_i is the standard error of firm i's
  CAR from its own estimation window (the engine's `se`, which includes the estimation error of the
  coefficients). Standardising before the cross-sectional test lets the event change the variance.
- Kolari and Pynnönen (2010, Review of Financial Studies 23(11), 3996-4025) adjust the BMP statistic for
  cross-sectional correlation of abnormal returns, which matters when events cluster in calendar time:

      t_KP = t_BMP sqrt((1 - rbar) / (1 + (N - 1) rbar)),

  with rbar the average pairwise correlation of the events' estimation-window residuals. The formula is
  as documented by eventstudytools.com ("Significance tests"); it has not been checked against the
  paper's text. `average_residual_correlation` computes rbar over the pairs whose estimation windows
  overlap by at least `min_overlap` days and counts non-overlapping pairs as zero correlation.
- Sign test: the share of positive CARs against 1/2 (exact binomial, two-sided).
"""

from __future__ import annotations

import itertools
import math
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import stats


def _clean(values) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return v[np.isfinite(v)]


def cross_sectional_t(cars) -> dict:
    v = _clean(cars)
    n = len(v)
    if n < 2:
        return {"n": n, "mean": float(v.mean()) if n else math.nan, "t": math.nan, "p": math.nan}
    sd = v.std(ddof=1)
    t = v.mean() / (sd / math.sqrt(n)) if sd > 0 else math.nan
    return {"n": n, "mean": float(v.mean()), "t": float(t), "p": float(2 * stats.t.sf(abs(t), n - 1))}


def bmp_test(cars, standard_errors) -> dict:
    cars, se = np.asarray(cars, dtype=float), np.asarray(standard_errors, dtype=float)
    keep = np.isfinite(cars) & np.isfinite(se) & (se > 0)
    scar = cars[keep] / se[keep]
    result = cross_sectional_t(scar)
    return {"n": result["n"], "mean_scar": result["mean"], "t": result["t"], "p": result["p"]}


def kolari_pynnonen(t_bmp: float, n: int, rbar: float) -> float:
    """The BMP statistic adjusted for an average cross-correlation rbar of N events."""
    if n < 2 or not np.isfinite(t_bmp):
        return math.nan
    denominator = 1 + (n - 1) * rbar
    if denominator <= 0 or rbar >= 1:
        raise ValueError("rbar must lie in (-1/(N-1), 1)")
    return float(t_bmp * math.sqrt((1 - rbar) / denominator))


def average_residual_correlation(residuals: Mapping[str, pd.Series], min_overlap: int = 30) -> dict:
    """Average pairwise correlation of estimation-window residuals (dated Series, one per event).

    Pairs with fewer than `min_overlap` common dates count as zero. Returns rbar, the number of pairs
    and the number of pairs with enough overlap.
    """
    keys = list(residuals)
    total, overlapping, pairs = 0.0, 0, 0
    for a, b in itertools.combinations(keys, 2):
        pairs += 1
        joined = pd.concat([residuals[a], residuals[b]], axis=1, join="inner").dropna()
        if len(joined) >= min_overlap:
            overlapping += 1
            total += float(joined.corr().iloc[0, 1])
    return {"rbar": total / pairs if pairs else math.nan, "pairs": pairs, "pairs_overlapping": overlapping}


def sign_test(cars) -> dict:
    v = _clean(cars)
    v = v[v != 0]
    n, positive = len(v), int((v > 0).sum())
    p = float(stats.binomtest(positive, n, 0.5).pvalue) if n else math.nan
    return {"n": n, "positive": positive, "share_positive": positive / n if n else math.nan, "p": p}


def summarise(cars, standard_errors, rbar: float = 0.0) -> dict:
    """All cross-sectional tests of one window, in one row."""
    cs, bmp, sign = cross_sectional_t(cars), bmp_test(cars, standard_errors), sign_test(cars)
    t_kp = kolari_pynnonen(bmp["t"], bmp["n"], rbar)
    return {
        "n": cs["n"],
        "mean_car": cs["mean"],
        "median_car": float(np.median(_clean(cars))) if cs["n"] else math.nan,
        "t_cross_sectional": cs["t"],
        "p_cross_sectional": cs["p"],
        "mean_scar": bmp["mean_scar"],
        "t_bmp": bmp["t"],
        "p_bmp": bmp["p"],
        "rbar": rbar,
        "t_kolari_pynnonen": t_kp,
        "p_kolari_pynnonen": float(2 * stats.t.sf(abs(t_kp), bmp["n"] - 1)) if np.isfinite(t_kp) else math.nan,
        "share_positive": sign["share_positive"],
        "p_sign": sign["p"],
    }
