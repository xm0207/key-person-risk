"""A general event-study engine: any ticker, any list of event dates, any set of windows.

Model. The market model of mega7-decomposition (`mega7.event_study`, written by the same author), with an
optional sector factor orthogonalised to the market:

    r_t = alpha + beta_m m_t + beta_s s*_t + e_t,      s*_t = s_t - (a + b m_t),

where a and b come from a regression of the sector return s on the market return m over the same
estimation window. Inside the estimation window s* is uncorrelated with m, so alpha and beta_m equal the
one-factor (market-only) estimates, and the one-factor abnormal return splits exactly into a sector part
beta_s s* and the company-specific part e. Each day's return is the sum of a market part (beta_m m), a
sector part (beta_s s*), alpha and the abnormal return. `fit_model` re-implements mega7's estimator so
that it can also return the residuals and (X'X)^-1; a test checks that it gives mega7's coefficients.

Estimation window. mega7's rule: `length` trading days (250) ending `gap` trading days (21) before the
first day of the event window. Here the event window is the union of all requested windows, so with a
pre-event window [-60, -2] the estimation window ends on day -81 and no pre-event day is used to
estimate the model. `anchor` fixes that first day explicitly instead (it may not come after the start of
any window). The calendar is the set of days on which the stock, the market and the sector (if any) all
have a return.

Windows are given in trading days relative to day 0. A window that runs past the end of the data is cut
there and flagged (`complete` is False).

Statistics for each window (L days, CAR the sum of the daily abnormal returns):

- z = CAR / (sigma sqrt(L)): mega7's scale, ignoring estimation error.
- t = CAR / se, se = sigma sqrt(L + 1' X_e (X'X)^-1 X_e' 1), the standard error of the sum of L
  out-of-sample OLS prediction errors (it adds the coefficients' estimation error to sigma^2 L); p from
  t with n - k degrees of freedom. This is the conventional parametric test, which assumes normal residuals.
- the SQ test of `keyperson.sq_test`: CAR against the sums of L consecutive estimation-window residuals.
- the same CAR, t and SQ p-value for the market-only model, whose abnormal return is the sector part
  plus the company-specific part.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from .sq_test import sq_test


@dataclass(frozen=True)
class Window:
    """An event window in trading days relative to day 0, inclusive at both ends."""

    name: str
    start: int
    end: int

    def __post_init__(self):
        if self.end < self.start:
            raise ValueError(f"window {self.name}: end {self.end} is before start {self.start}")

    @property
    def length(self) -> int:
        return self.end - self.start + 1


DEFAULT_WINDOWS = (
    Window("pre_60_2", -60, -2),
    Window("pre_20_2", -20, -2),
    Window("day_0", 0, 0),
    Window("ann_0_1", 0, 1),
    Window("ann_m1_1", -1, 1),
    Window("post_2_20", 2, 20),
    Window("post_2_60", 2, 60),
)


@dataclass(frozen=True)
class EstimationRule:
    """`length` trading days ending `gap` days before the anchor day (relative to day 0).

    `anchor` None means the first day of the earliest window. `min_observations` None means `length`:
    an event without a full estimation window is skipped.
    """

    length: int = 250
    gap: int = 21
    min_observations: int | None = None
    anchor: int | None = None

    def __post_init__(self):
        if self.length < 10:
            raise ValueError("length must be at least 10")
        if self.gap < 1:
            raise ValueError("gap must be at least 1, so that no event day is in its own estimation window")
        if self.min_observations is not None and not 10 <= self.min_observations <= self.length:
            raise ValueError("min_observations must be between 10 and length")

    @property
    def required(self) -> int:
        return self.length if self.min_observations is None else self.min_observations


@dataclass(frozen=True)
class ModelFit:
    """One estimate of the market model (with or without the orthogonalised sector factor).

    `xtx_inv` is (X'X)^-1 of the regressors [1, m, s*] (or [1, m]); `residuals` are the estimation-window
    residuals, dated by `residual_dates`. The sector fields are NaN without a sector.
    """

    alpha: float
    beta_market: float
    beta_sector: float
    sector_intercept: float
    sector_beta_to_market: float
    sigma: float
    r_squared: float
    observations: int
    parameters: int
    xtx_inv: np.ndarray = field(repr=False)
    residuals: np.ndarray = field(repr=False)
    residual_dates: pd.DatetimeIndex = field(repr=False)

    @property
    def has_sector(self) -> bool:
        return not math.isnan(self.beta_sector)

    @property
    def degrees_of_freedom(self) -> int:
        return self.observations - self.parameters

    def design(self, m: np.ndarray, s: np.ndarray | None) -> np.ndarray:
        """Regressors for out-of-sample days, with the sector orthogonalised by this fit's a and b."""
        m = np.asarray(m, dtype=float)
        columns = [np.ones(len(m)), m]
        if self.has_sector:
            if s is None:
                raise ValueError("a fit with a sector factor needs the sector's returns")
            columns.append(np.asarray(s, dtype=float) - self.sector_intercept - self.sector_beta_to_market * m)
        return np.column_stack(columns)

    @property
    def coefficients(self) -> np.ndarray:
        values = [self.alpha, self.beta_market] + ([self.beta_sector] if self.has_sector else [])
        return np.array(values)


def fit_model(r, m, s=None, dates=None) -> ModelFit:
    """OLS of r on a constant, m and (if given) s orthogonalised to m, as in mega7.event_study._estimate."""
    r = np.asarray(r, dtype=float)
    m = np.asarray(m, dtype=float)
    n = len(r)
    k = 2 if s is None else 3
    if n <= k + 1:
        raise ValueError("not enough observations for the market model")
    base = np.column_stack([np.ones(n), m])
    a = b = beta_s = math.nan
    x = base
    if s is not None:
        s = np.asarray(s, dtype=float)
        (a, b), *_ = np.linalg.lstsq(base, s, rcond=None)
        x = np.column_stack([base, s - a - b * m])
    coef, *_ = np.linalg.lstsq(x, r, rcond=None)
    resid = r - x @ coef
    if s is not None:
        beta_s = float(coef[2])
    sigma = math.sqrt(resid @ resid / (n - k))
    r2 = 1.0 - resid @ resid / np.sum((r - r.mean()) ** 2)
    dates = pd.DatetimeIndex(dates) if dates is not None else pd.RangeIndex(n)
    return ModelFit(
        float(coef[0]), float(coef[1]), float(beta_s), float(a), float(b), sigma, float(r2), n, k,
        np.linalg.inv(x.T @ x), resid, dates,
    )


def window_statistics(fit: ModelFit, r: np.ndarray, m: np.ndarray, s: np.ndarray | None) -> dict:
    """CAR and its tests over the days given (r, m, s are the window's returns)."""
    x = fit.design(m, s)
    abnormal = np.asarray(r, dtype=float) - x @ fit.coefficients
    length = len(abnormal)
    car = float(abnormal.sum())
    ones = x.sum(axis=0)  # 1' X_e
    se = fit.sigma * math.sqrt(length + float(ones @ fit.xtx_inv @ ones))
    t = car / se
    sq = sq_test(car, fit.residuals, length) if length <= len(fit.residuals) else None
    out = {
        "car": car,
        "z": car / (fit.sigma * math.sqrt(length)),
        "se": se,
        "t": t,
        "p_t": float(2 * stats.t.sf(abs(t), fit.degrees_of_freedom)),
        "sq_p_lower": sq.p_lower if sq else math.nan,
        "sq_p_upper": sq.p_upper if sq else math.nan,
        "sq_p_two_sided": sq.p_two_sided if sq else math.nan,
        "sq_reference_size": sq.reference_size if sq else 0,
        "market_part": fit.beta_market * float(np.sum(m)),
        "alpha_part": fit.alpha * length,
    }
    out["sector_part"] = fit.beta_sector * float(x[:, 2].sum()) if fit.has_sector else 0.0
    return out


@dataclass
class EventResult:
    """Everything the engine produced for one event: status, fits and one row per window."""

    event_id: str
    ticker: str
    sector: str | None
    status: str
    day0_requested: pd.Timestamp
    day0: pd.Timestamp | None = None
    fit: ModelFit | None = None
    fit_market_only: ModelFit | None = None
    windows: pd.DataFrame = field(default_factory=pd.DataFrame)

    def estimation_row(self) -> dict:
        row = {"event_id": self.event_id, "ticker": self.ticker, "sector": self.sector or "none",
               "status": self.status, "day0_requested": self.day0_requested.date(),
               "day0": self.day0.date() if self.day0 is not None else None}
        if self.fit is not None:
            f, f1 = self.fit, self.fit_market_only
            row |= {
                "estimation_start": f.residual_dates[0].date(),
                "estimation_end": f.residual_dates[-1].date(),
                "observations": f.observations,
                "alpha_daily": f.alpha,
                "beta_market": f.beta_market,
                "beta_sector": f.beta_sector,
                "sector_intercept": f.sector_intercept,
                "sector_beta_to_market": f.sector_beta_to_market,
                "sigma_daily": f.sigma,
                "r_squared": f.r_squared,
                "sigma_daily_market_only": f1.sigma,
                "r_squared_market_only": f1.r_squared,
            }
        return row


class EventStudyEngine:
    """Runs the market model around any number of events on one panel of daily returns.

    Why a class: it owns the return panel, the market series, the windows and the estimation rule, so
    every event is measured on the same calendar logic with the same rule. Public interface:

    - `run_event(event_id, ticker, day0, sector=None)` one event; `sector` is a column of `returns`,
      a Series (for example an ex-firm basket) or None for the market-only model.
    - `run(events)` a table of events (columns event_id, ticker, day0 and optionally sector) and
      returns (windows table, estimation table, list of EventResult).

    The returns may come from any source (Yahoo via mega7's PriceStore, CRSP, ...): a DataFrame of
    simple daily returns indexed by date, one column per security, NaN where there is no return.
    """

    def __init__(self, returns: pd.DataFrame, market: str, rule: EstimationRule = EstimationRule(),
                 windows: Iterable[Window] = DEFAULT_WINDOWS):
        if market not in returns:
            raise KeyError(f"market series {market!r} is not in the returns")
        self.returns = returns.sort_index()
        self.market = market
        self.rule = rule
        self.windows = tuple(windows)
        if not self.windows:
            raise ValueError("at least one window is needed")
        names = [w.name for w in self.windows]
        if len(set(names)) != len(names):
            raise ValueError("window names must be unique")
        self.first_window_day = min(w.start for w in self.windows)
        self.anchor = self.first_window_day if rule.anchor is None else rule.anchor
        if self.anchor > self.first_window_day:
            raise ValueError("the anchor comes after the start of a window, so the estimation window could overlap it")

    def _aligned(self, ticker: str, sector) -> tuple[pd.DataFrame, str | None]:
        columns = {"stock": self.returns[ticker], "market": self.returns[self.market]}
        label = None
        if sector is not None:
            if isinstance(sector, str):
                columns["sector"], label = self.returns[sector], sector
            else:
                columns["sector"], label = sector, getattr(sector, "name", None) or "custom"
        return pd.concat(columns, axis=1, join="inner").dropna(), label

    def run_event(self, event_id: str, ticker: str, day0, sector=None) -> EventResult:
        day0 = pd.Timestamp(day0)
        if ticker not in self.returns:
            return EventResult(event_id, ticker, sector if isinstance(sector, str) else None, "no_data", day0)
        data, label = self._aligned(ticker, sector)
        result = EventResult(event_id, ticker, label, "ok", day0)
        p0 = int(data.index.searchsorted(day0))
        if p0 >= len(data):
            result.status = "after_data_end"
            return result
        result.day0 = data.index[p0]
        est_end = p0 + self.anchor - self.rule.gap
        est_start = max(0, est_end - self.rule.length + 1)
        if est_end < 0 or est_end - est_start + 1 < self.rule.required:
            result.status = "insufficient_history"
            return result
        est = data.iloc[est_start: est_end + 1]
        s_est = est["sector"].to_numpy() if "sector" in est else None
        result.fit = fit_model(est["stock"], est["market"], s_est, est.index)
        result.fit_market_only = fit_model(est["stock"], est["market"], None, est.index)

        rows = []
        for w in self.windows:
            lo, hi = p0 + w.start, p0 + w.end
            lo_c, hi_c = max(lo, 0), min(hi, len(data) - 1)
            row = {"event_id": event_id, "ticker": ticker, "sector": label or "none", "window": w.name,
                   "rel_start": w.start, "rel_end": w.end, "days_expected": w.length}
            if hi_c < lo_c:
                rows.append(row | {"days": 0, "complete": False})
                continue
            rows_w = data.iloc[lo_c: hi_c + 1]
            r, m = rows_w["stock"].to_numpy(), rows_w["market"].to_numpy()
            s = rows_w["sector"].to_numpy() if "sector" in rows_w else None
            two = window_statistics(result.fit, r, m, s)
            one = window_statistics(result.fit_market_only, r, m, None)
            row |= {
                "date_start": rows_w.index[0].date(),
                "date_end": rows_w.index[-1].date(),
                "days": len(rows_w),
                "complete": len(rows_w) == w.length,
                "sum_of_returns": float(r.sum()),
                "bhr": float(np.prod(1 + r) - 1),
                "market_bhr": float(np.prod(1 + m) - 1),
            }
            row |= two
            row |= {
                "car_market_only": one["car"],
                "z_market_only": one["z"],
                "t_market_only": one["t"],
                "p_t_market_only": one["p_t"],
                "sq_p_two_sided_market_only": one["sq_p_two_sided"],
            }
            rows.append(row)
        result.windows = pd.DataFrame(rows)
        return result

    def run(self, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[EventResult]]:
        """Run every event in `events` (columns event_id, ticker, day0 and optionally sector)."""
        results = []
        for row in events.itertuples(index=False):
            sector = getattr(row, "sector", None)
            sector = None if sector is None or (isinstance(sector, float) and math.isnan(sector)) or sector == "" else sector
            results.append(self.run_event(row.event_id, row.ticker, row.day0, sector))
        windows = [r.windows for r in results if not r.windows.empty]
        windows_table = pd.concat(windows, ignore_index=True) if windows else pd.DataFrame()
        estimation_table = pd.DataFrame([r.estimation_row() for r in results])
        return windows_table, estimation_table, results
