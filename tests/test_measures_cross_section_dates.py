import math

import numpy as np
import pandas as pd
import pytest

from keyperson.cross_section import (
    average_residual_correlation,
    bmp_test,
    cross_sectional_t,
    kolari_pynnonen,
    sign_test,
    summarise,
)
from keyperson.dates import day_zero
from keyperson.measures import (
    aggregate_post_event_ratio,
    aggregate_priced_in_share,
    event_profile,
    post_event_ratio,
    priced_in_share,
)

# --- day 0 -------------------------------------------------------------------------------------------

DAYS = pd.DatetimeIndex(["2026-11-05", "2026-11-06", "2026-11-09", "2026-11-10"])


def test_day_zero_rules():
    assert day_zero(DAYS, "2026-11-06", "before_open") == pd.Timestamp("2026-11-06")
    assert day_zero(DAYS, "2026-11-06", "during_hours") == pd.Timestamp("2026-11-06")
    assert day_zero(DAYS, "2026-11-06", "after_close") == pd.Timestamp("2026-11-09")
    assert day_zero(DAYS, "2026-11-08", "non_trading_day") == pd.Timestamp("2026-11-09")
    assert day_zero(DAYS, "2026-11-06", "unknown") == pd.Timestamp("2026-11-06")
    assert day_zero(DAYS, "2026-11-07", "before_open") == pd.Timestamp("2026-11-09")  # a Saturday


def test_day_zero_rejects_bad_labels():
    with pytest.raises(ValueError):
        day_zero(DAYS, "2026-11-06", "non_trading_day")
    with pytest.raises(ValueError):
        day_zero(DAYS, "2026-11-06", "evening")
    with pytest.raises(ValueError):
        day_zero(DAYS, "2026-11-10", "after_close")


# --- priced-in share and post-event drift -------------------------------------------------------------


def test_priced_in_share_values():
    assert priced_in_share(-0.02, -0.08) == pytest.approx(0.2)
    assert priced_in_share(0.0, -0.05) == pytest.approx(0.0)
    assert priced_in_share(0.02, -0.08) == pytest.approx(0.02 / -0.06)  # opposite pre-move: negative share
    assert math.isnan(priced_in_share(0.001, -0.001, min_total=0.01))


def test_post_event_ratio_values():
    assert post_event_ratio(-0.03, -0.06) == pytest.approx(0.5)  # continuation
    assert post_event_ratio(0.03, -0.06) == pytest.approx(-0.5)  # reversal
    assert math.isnan(post_event_ratio(0.01, 0.0001, min_announcement=0.001))


def test_aggregates_are_sign_aligned():
    pre = np.array([-0.02, 0.01])
    ann = np.array([-0.08, 0.09])
    # totals -0.10 and +0.10: aligned pre moves 0.02 and 0.01 out of 0.20
    assert aggregate_priced_in_share(pre, ann) == pytest.approx(0.15)
    post = np.array([0.04, -0.03])  # both reverse
    assert aggregate_post_event_ratio(post, ann) == pytest.approx(-0.07 / 0.17)
    assert math.isnan(aggregate_priced_in_share([np.nan], [np.nan]))


def test_event_profile_pivots_the_engine_table():
    rows = []
    for event, values in {"a": (-0.02, -0.08, 0.01), "b": (0.0, 0.05, 0.02)}.items():
        for window, value in zip(["pre", "ann", "post"], values):
            rows.append({"event_id": event, "ticker": "X", "sector": "S", "window": window, "car": value,
                         "complete": not (event == "b" and window == "post")})
    profile = event_profile(pd.DataFrame(rows), "pre", "ann", "post").set_index("event_id")
    assert profile.loc["a", "priced_in_share"] == pytest.approx(0.2)
    assert profile.loc["a", "post_event_ratio"] == pytest.approx(-0.125)
    assert math.isnan(profile.loc["b", "car_post"])  # incomplete window


# --- cross-sectional tests ----------------------------------------------------------------------------


def test_cross_sectional_t_by_hand():
    v = np.array([0.01, 0.03, -0.01, 0.05])
    out = cross_sectional_t(v)
    assert out["t"] == pytest.approx(v.mean() / (v.std(ddof=1) / 2))
    assert out["n"] == 4


def test_bmp_is_the_t_test_of_standardised_cars():
    cars = np.array([0.02, -0.01, 0.04, 0.03])
    se = np.array([0.01, 0.02, 0.02, 0.01])
    assert bmp_test(cars, se)["t"] == pytest.approx(cross_sectional_t(cars / se)["t"])


def test_kolari_pynnonen_adjustment():
    assert kolari_pynnonen(3.0, 20, 0.0) == pytest.approx(3.0)
    assert kolari_pynnonen(3.0, 20, 0.05) == pytest.approx(3.0 * math.sqrt(0.95 / 1.95))
    with pytest.raises(ValueError):
        kolari_pynnonen(3.0, 3, -0.6)


def test_average_residual_correlation():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2020-01-01", periods=200)
    common = rng.normal(size=200)
    a = pd.Series(common + rng.normal(size=200), index=dates)
    b = pd.Series(common + rng.normal(size=200), index=dates)
    far = pd.Series(rng.normal(size=200), index=pd.bdate_range("2010-01-01", periods=200))
    out = average_residual_correlation({"a": a, "b": b, "far": far})
    assert out["pairs"] == 3 and out["pairs_overlapping"] == 1
    assert out["rbar"] == pytest.approx(a.corr(b) / 3)


def test_sign_test_and_summary():
    cars = np.array([-0.05, -0.02, -0.03, 0.01, -0.04])
    se = np.full(5, 0.02)
    s = sign_test(cars)
    assert s["positive"] == 1 and s["p"] == pytest.approx(0.375)
    row = summarise(cars, se, rbar=0.0)
    assert row["t_kolari_pynnonen"] == pytest.approx(row["t_bmp"])
    assert row["n"] == 5
