import math

import numpy as np
import pandas as pd
import pytest

from mega7.event_study import EventStudy, fit_market_model
from keyperson.engine import DEFAULT_WINDOWS, EstimationRule, EventStudyEngine, Window, fit_model


def test_fit_model_matches_mega7(panel):
    rows = panel.iloc[100:350]
    for sector in (None, rows["SEC"]):
        ours = fit_model(rows["STOCK"], rows["MKT"], None if sector is None else sector.to_numpy(), rows.index)
        theirs = fit_market_model(rows["STOCK"], rows["MKT"], sector)
        assert ours.alpha == pytest.approx(theirs.alpha, abs=1e-14)
        assert ours.beta_market == pytest.approx(theirs.beta_market, rel=1e-12)
        assert ours.sigma == pytest.approx(theirs.residual_sd, rel=1e-12)
        assert ours.r_squared == pytest.approx(theirs.r_squared, rel=1e-12)
        if sector is not None:
            assert ours.beta_sector == pytest.approx(theirs.beta_sector, rel=1e-12)
            assert ours.sector_beta_to_market == pytest.approx(theirs.sector_beta_to_market, rel=1e-12)


def test_fit_recovers_the_true_betas(panel):
    fit = fit_model(panel["STOCK"], panel["MKT"], panel["SEC"].to_numpy(), panel.index)
    assert fit.beta_market == pytest.approx(1.2, abs=0.1)
    assert fit.beta_sector == pytest.approx(0.5, abs=0.1)
    assert fit.sector_beta_to_market == pytest.approx(1.4, abs=0.05)


def test_engine_reproduces_mega7_event_study(panel):
    """With the anchor set to mega7's first event-window day, the engine's day-0 CAR and window totals
    equal mega7's EventStudy, including its estimation window."""
    day0 = panel.index[600]
    first = panel.index[597]  # mega7's first day of the event window = day -3
    windows = (Window("day_0", 0, 0), Window("pre_3_1", -3, -1), Window("ann_0_1", 0, 1))
    engine = EventStudyEngine(panel, "MKT", EstimationRule(anchor=-3), windows)
    result = engine.run_event("e1", "STOCK", day0, "SEC")
    study = EventStudy(panel["STOCK"], panel["MKT"], panel["SEC"])
    est = study.estimation_window(first)
    assert result.fit.residual_dates[0] == est[0] and result.fit.residual_dates[-1] == est[-1]
    day = study.days(first, [day0]).iloc[0]
    row = result.windows.set_index("window").loc["day_0"]
    assert row["car"] == pytest.approx(day["abnormal"], abs=1e-15)
    assert row["z"] == pytest.approx(day["z"], rel=1e-12)
    assert row["sector_part"] == pytest.approx(day["sector"], abs=1e-15)
    assert row["market_part"] == pytest.approx(day["market"], abs=1e-15)
    total = study.window(first, panel.index[597], panel.index[599])
    pre = result.windows.set_index("window").loc["pre_3_1"]
    assert pre["car"] == pytest.approx(total["abnormal"], abs=1e-15)
    assert pre["z"] == pytest.approx(total["z"], rel=1e-12)


def test_parts_add_up_and_one_factor_splits(panel):
    engine = EventStudyEngine(panel, "MKT")
    result = engine.run_event("e1", "STOCK", panel.index[700], "SEC")
    w = result.windows
    total = w["market_part"] + w["sector_part"] + w["alpha_part"] + w["car"]
    assert np.allclose(total, w["sum_of_returns"], atol=1e-14)
    assert np.allclose(w["car_market_only"], w["sector_part"] + w["car"], atol=1e-14)


def test_default_estimation_window_ends_gap_days_before_the_earliest_window(panel):
    engine = EventStudyEngine(panel, "MKT")
    p0 = 700
    result = engine.run_event("e1", "STOCK", panel.index[p0], "SEC")
    first = min(w.start for w in DEFAULT_WINDOWS)  # -60
    assert result.fit.residual_dates[-1] == panel.index[p0 + first - 21]
    assert result.fit.observations == 250
    rows = result.windows.set_index("window")
    assert rows.loc["pre_60_2", "date_start"] == panel.index[p0 - 60].date()
    assert rows.loc["post_2_60", "date_end"] == panel.index[p0 + 60].date()


def test_anchor_after_a_window_start_is_refused(panel):
    with pytest.raises(ValueError):
        EventStudyEngine(panel, "MKT", EstimationRule(anchor=0), (Window("pre", -5, -1),))


def test_standard_error_is_invariant_to_the_orthogonalisation(panel):
    """se uses (X'X)^-1 of [1, m, s*]; the same prediction-error variance follows from raw [1, m, s]."""
    engine = EventStudyEngine(panel, "MKT", windows=(Window("w", 0, 4),))
    p0 = 500
    result = engine.run_event("e1", "STOCK", panel.index[p0], "SEC")
    fit = result.fit
    est = panel.loc[fit.residual_dates]
    x = np.column_stack([np.ones(len(est)), est["MKT"], est["SEC"]])
    ev = panel.iloc[p0: p0 + 5]
    xe = np.column_stack([np.ones(5), ev["MKT"], ev["SEC"]])
    ones = xe.sum(axis=0)
    se = fit.sigma * math.sqrt(5 + ones @ np.linalg.inv(x.T @ x) @ ones)
    assert result.windows.loc[0, "se"] == pytest.approx(se, rel=1e-10)
    assert result.windows.loc[0, "se"] > fit.sigma * math.sqrt(5)


def test_a_planted_jump_is_recovered_and_is_extreme_for_the_sq_test(panel):
    data = panel.copy()
    p0 = 650
    engine = EventStudyEngine(data, "MKT")
    base = engine.run_event("e", "STOCK", data.index[p0], "SEC").windows.set_index("window").loc["day_0", "car"]
    data.iloc[p0, data.columns.get_loc("STOCK")] -= 0.15
    row = EventStudyEngine(data, "MKT").run_event("e", "STOCK", data.index[p0], "SEC").windows.set_index("window").loc["day_0"]
    assert row["car"] == pytest.approx(base - 0.15, abs=1e-12)
    assert row["sq_p_lower"] == pytest.approx(1 / 251)
    assert row["p_t"] < 1e-6


def test_statuses(panel):
    engine = EventStudyEngine(panel, "MKT")
    assert engine.run_event("a", "STOCK", panel.index[100], "SEC").status == "insufficient_history"
    assert engine.run_event("b", "STOCK", panel.index[-1] + pd.Timedelta(days=5)).status == "after_data_end"
    assert engine.run_event("c", "MISSING", panel.index[500]).status == "no_data"
    short = EventStudyEngine(panel, "MKT", EstimationRule(min_observations=200))
    assert short.run_event("d", "STOCK", panel.index[310], "SEC").status == "ok"


def test_windows_cut_by_the_end_of_the_data_are_flagged(panel):
    engine = EventStudyEngine(panel, "MKT")
    result = engine.run_event("e", "STOCK", panel.index[-10], "SEC")
    rows = result.windows.set_index("window")
    assert bool(rows.loc["ann_m1_1", "complete"])
    assert not bool(rows.loc["post_2_20", "complete"])
    assert rows.loc["post_2_20", "days"] == 8
    assert rows.loc["post_2_60", "days"] == 8


def test_day0_off_the_calendar_moves_to_the_next_trading_day(panel):
    engine = EventStudyEngine(panel, "MKT")
    saturday = pd.Timestamp("2017-06-03")
    result = engine.run_event("e", "STOCK", saturday, "SEC")
    assert result.day0 == pd.Timestamp("2017-06-05")


def test_market_only_model_without_sector(panel):
    engine = EventStudyEngine(panel, "MKT")
    result = engine.run_event("e", "OTHER", panel.index[700])
    w = result.windows
    assert (w["sector_part"] == 0).all()
    assert np.allclose(w["car"], w["car_market_only"])
    assert math.isnan(result.fit.beta_sector)


def test_run_on_a_table_with_series_sector(panel):
    engine = EventStudyEngine(panel, "MKT")
    events = pd.DataFrame({"event_id": ["a", "b"], "ticker": ["STOCK", "OTHER"],
                           "day0": [panel.index[600], panel.index[650]], "sector": ["SEC", None]})
    windows, estimation, results = engine.run(events)
    assert set(windows["event_id"]) == {"a", "b"}
    assert list(estimation["status"]) == ["ok", "ok"]
    assert estimation.loc[1, "sector"] == "none"
    custom = engine.run_event("c", "STOCK", panel.index[600], panel["SEC"].rename("SEC copy"))
    assert custom.windows["car"].to_numpy() == pytest.approx(windows[windows.event_id == "a"]["car"].to_numpy())
