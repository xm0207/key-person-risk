import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def panel():
    """Synthetic daily returns: a market, a sector that loads on it, and two stocks with known betas.

    STOCK = 0.0002 + 1.2 m + 0.5 (sector beyond the market) + e, with t(4) noise. No jumps.
    """
    rng = np.random.default_rng(7)
    n = 900
    dates = pd.bdate_range("2015-01-01", periods=n)
    m = rng.normal(0.0004, 0.01, n)
    sector_own = rng.normal(0, 0.008, n)
    s = 0.0001 + 1.4 * m + sector_own
    e = 0.012 * rng.standard_t(4, n) / np.sqrt(2)
    stock = 0.0002 + 1.2 * m + 0.5 * sector_own + e
    other = 0.0001 + 0.9 * m + rng.normal(0, 0.015, n)
    return pd.DataFrame({"MKT": m, "SEC": s, "STOCK": stock, "OTHER": other}, index=dates)
