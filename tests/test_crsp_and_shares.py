"""Delisting returns, the returns panel, and the ratio-of-means priced-in share."""
import numpy as np
import pandas as pd

from keyperson import crsp, measures


def test_add_delisting_combines_or_fills():
    dsf = pd.DataFrame({"permno": [1, 1, 2], "date": pd.to_datetime(["2010-01-04", "2010-01-05", "2010-01-04"]),
                        "ret": [0.01, -0.10, np.nan]})
    dl = pd.DataFrame({"permno": [1, 2], "dlstdt": ["2010-01-05", "2010-01-04"], "dlret": [-0.5, -0.3]})
    out = crsp.add_delisting(dsf, dl).set_index(["permno", "date"])["ret"]
    assert np.isclose(out[(1, pd.Timestamp("2010-01-05"))], 0.9 * 0.5 - 1)
    assert np.isclose(out[(2, pd.Timestamp("2010-01-04"))], -0.3)


def test_returns_panel_adds_market():
    dsf = pd.DataFrame({"permno": [7], "date": pd.to_datetime(["2010-01-04"]), "ret": [0.02]})
    ff = pd.DataFrame({"date": ["2010-01-04"], "mktrf": [0.01], "rf": [0.0001]})
    p = crsp.returns_panel(dsf, ff)
    assert list(p.columns) == ["7", "MKT"] and np.isclose(p.loc["2010-01-04", "MKT"], 0.0101)


def test_share_of_mean_move_lets_noise_cancel():
    rng = np.random.default_rng(0)
    pre = rng.normal(0, 0.10, 5000)          # noisy pre-event window, nothing anticipated
    ann = rng.normal(-0.03, 0.02, 5000)      # the announcement carries all of the move
    assert abs(measures.share_of_mean_move(pre, ann)) < 0.15
    assert measures.aggregate_priced_in_share(pre, ann) > 0.5  # the sign-aligned version is fooled by noise
    assert np.isclose(measures.ratio_of_means([1, 3], [2, 2]), 1.0)
