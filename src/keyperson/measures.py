"""How much of an event's price impact comes before, at and after the announcement.

For each event, with CARs over a pre-event window (for example [-20, -2]), an announcement window
([-1, +1]) and a post-event window ([+2, +20]):

- priced-in share = CAR_pre / (CAR_pre + CAR_ann): the part of the move up to the end of the announcement
  window that had already happened before it. 0 means none of it was anticipated, 1 means all of it. It
  is negative when the stock moved the other way before the announcement, and above 1 when the
  announcement partly reversed a pre-event move. It is undefined (NaN) when |CAR_pre + CAR_ann| is
  below `min_total`, because a ratio with a denominator near zero says nothing.
- post-event ratio = CAR_post / CAR_ann: positive means drift in the direction of the announcement
  reaction (continuation), negative means reversal. NaN when |CAR_ann| is below `min_announcement`.

Per-event ratios are noisy, so the cross-section is better summarised by sign-aligned ratios of sums,
which weight events by the size of their move and cannot be dominated by a denominator near zero:

    aggregate priced-in share = sum_i d_i CAR_pre,i / sum_i |CAR_pre,i + CAR_ann,i|,   d_i = sign(CAR_pre,i + CAR_ann,i)
    aggregate post-event ratio = sum_i a_i CAR_post,i / sum_i |CAR_ann,i|,              a_i = sign(CAR_ann,i)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def priced_in_share(pre, announcement, min_total: float = 0.0):
    """CAR_pre / (CAR_pre + CAR_ann), NaN where |CAR_pre + CAR_ann| < min_total (element-wise)."""
    pre, ann = np.asarray(pre, dtype=float), np.asarray(announcement, dtype=float)
    total = pre + ann
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(np.abs(total) >= max(min_total, 1e-15), pre / total, np.nan)
    return share if share.ndim else float(share)


def post_event_ratio(post, announcement, min_announcement: float = 0.0):
    """CAR_post / CAR_ann, NaN where |CAR_ann| < min_announcement (element-wise)."""
    post, ann = np.asarray(post, dtype=float), np.asarray(announcement, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(ann) >= max(min_announcement, 1e-15), post / ann, np.nan)
    return ratio if ratio.ndim else float(ratio)


def aggregate_priced_in_share(pre, announcement) -> float:
    """Sign-aligned share of the total pre-plus-announcement move that came before the announcement."""
    pre, ann = np.asarray(pre, dtype=float), np.asarray(announcement, dtype=float)
    keep = np.isfinite(pre) & np.isfinite(ann)
    total = pre[keep] + ann[keep]
    denominator = np.abs(total).sum()
    return float(np.sum(np.sign(total) * pre[keep]) / denominator) if denominator > 0 else float("nan")


def aggregate_post_event_ratio(post, announcement) -> float:
    """Sign-aligned post-event drift relative to the announcement reaction (negative = reversal)."""
    post, ann = np.asarray(post, dtype=float), np.asarray(announcement, dtype=float)
    keep = np.isfinite(post) & np.isfinite(ann)
    denominator = np.abs(ann[keep]).sum()
    return float(np.sum(np.sign(ann[keep]) * post[keep]) / denominator) if denominator > 0 else float("nan")


def event_profile(windows: pd.DataFrame, pre: str, announcement: str, post: str, value: str = "car",
                  min_total: float = 0.0, min_announcement: float = 0.0) -> pd.DataFrame:
    """One row per event: the CARs of the three windows, the priced-in share and the post-event ratio.

    `windows` is the engine's windows table; `pre`, `announcement` and `post` are window names.
    Windows that are incomplete (cut by the end of the data) give NaN.
    """
    keep = windows[windows["window"].isin([pre, announcement, post])].copy()
    keep.loc[~keep["complete"].astype(bool), value] = np.nan
    wide = keep.pivot_table(index=["event_id", "ticker", "sector"], columns="window", values=value, aggfunc="first")
    wide = wide.reindex(columns=[pre, announcement, post])
    out = pd.DataFrame(
        {
            f"{value}_pre": wide[pre],
            f"{value}_announcement": wide[announcement],
            f"{value}_post": wide[post],
        },
        index=wide.index,
    )
    out["priced_in_share"] = priced_in_share(out[f"{value}_pre"], out[f"{value}_announcement"], min_total)
    out["post_event_ratio"] = post_event_ratio(out[f"{value}_post"], out[f"{value}_announcement"], min_announcement)
    return out.reset_index()


def share_of_mean_move(pre, announcement) -> float:
    """mean(CAR_pre) / (mean(CAR_pre) + mean(CAR_ann)): the share of the average move that came first.

    Preferred to `aggregate_priced_in_share` for large samples of noisy events. The sign-aligned version
    weights each event by |CAR_pre + CAR_ann|, and when the pre-event window is much longer (so much
    noisier) than the announcement window that sum is mostly pre-event noise, which pushes the share
    towards 1 even when nothing is anticipated: CEO retirements, with mean CARs near zero, score 0.97 on
    [-120, -2]. Averaging first lets the noise cancel. NaN when the mean total is exactly zero.
    """
    pre, ann = np.asarray(pre, dtype=float), np.asarray(announcement, dtype=float)
    keep = np.isfinite(pre) & np.isfinite(ann)
    total = pre[keep].mean() + ann[keep].mean()
    return float(pre[keep].mean() / total) if total != 0 else float("nan")


def ratio_of_means(numerator, denominator) -> float:
    """mean(numerator) / mean(denominator) over events where both are finite (for example post / announcement)."""
    a, b = np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    d = b[keep].mean()
    return float(a[keep].mean() / d) if d != 0 else float("nan")
