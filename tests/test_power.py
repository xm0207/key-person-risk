"""Offline tests of keyperson.power on hand-made ExecuComp and Compustat rows."""
import numpy as np
import pandas as pd

from keyperson import power


def events():
    return pd.DataFrame({"gvkey": ["000001"], "fyear": [2010], "co_per_rol": [11], "surname": ["Lee"],
                         "ref_date": [pd.Timestamp("2010-06-01")], "event_date": [pd.Timestamp("2010-06-01")],
                         "rp_entity_id": ["E1"]})


def anncomp():
    rows = [  # gvkey, year, execid, co_per_rol, ceoann, title, tdc1, becameceo
        ("000001", 2009, "X", 11, "CEO", "Chairman & CEO", 600.0, "2004-06-01"),
        ("000001", 2009, "Y", 12, None, "CFO", 200.0, None),
        ("000001", 2009, "Z", 13, None, "COO", 200.0, None),
        ("000002", 2007, "X", 21, None, "EVP", 100.0, None),   # earlier job elsewhere
        ("000003", 2013, "X", 31, "CEO", "CEO", 300.0, None),  # later CEO elsewhere
    ]
    a = pd.DataFrame(rows, columns=["gvkey", "year", "execid", "co_per_rol", "ceoann", "title", "tdc1", "becameceo"])
    return a.assign(age=55, shrown_excl_opts_pct=1.5)


def test_ceo_power_from_last_full_year():
    out = power.ceo_power(events(), anncomp()).iloc[0]
    assert out["power_year"] == 2009 and out["execid"] == "X" and out["duality"]
    assert np.isclose(out["pay_slice"], 0.6) and np.isclose(out["tenure_years"], 6.0, atol=0.01)


def test_network_and_career():
    ev = power.ceo_power(events(), anncomp())
    out = power.network_and_career(ev, anncomp()).iloc[0]
    assert out["prior_firms"] == 1 and out["later_exec_5y"] == 1.0 and out["later_ceo_5y"] == 1.0


def test_firm_power_sales_share_within_sic2():
    funda = pd.DataFrame({"gvkey": ["000001", "000009"], "fyear": [2009, 2009], "at": [100.0, 50.0],
                          "sale": [30.0, 90.0], "csho": [10.0, 1.0], "prcc_f": [5.0, 1.0], "sich": [2834.0, 2836.0]})
    out = power.firm_power(events(), funda).iloc[0]
    assert np.isclose(out["sales_share"], 0.25) and np.isclose(out["log_mv"], np.log(50.0))
