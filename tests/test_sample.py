"""Offline tests of the event-sample builder (keyperson.sample) on small hand-made frames."""
import pandas as pd

from keyperson import sample


def test_surname_strips_suffixes():
    assert sample.surname("John A. Smith Jr.") == "Smith"
    assert sample.surname("Mary O'Neil, Ph.D.") == "O'Neil"
    assert sample.surname("Lee III") == "Lee"


def gentry_frame():
    return pd.DataFrame({
        "dismissal_dataset_id": [1, 2, 3], "coname": ["A", "A", "B"], "gvkey": [1004, 1004, 2000],
        "fyear": [2010, 1998, 2012], "co_per_rol": [1, 2, 3],
        "exec_fullname": ["Ann Lee", "Bob Old", "Carl Diaz Jr."], "departure_code": [4, 5, 7],
        "interim_coceo": [None, None, None],
        "left_ceo_office": ["2010-06-30", "1998-01-01", "2012-01-01"],
        "date_departure_announced": ["2010-06-01", None, None]})


def test_load_gentry_keeps_codes_1_to_6_from_2000():
    g = sample.load_gentry(gentry_frame())
    assert list(g["gvkey"]) == ["001004"] and g["reason"].iloc[0] == "dismissed_misconduct"
    assert g["ref_date"].iloc[0] == pd.Timestamp("2010-06-01") and g["surname"].iloc[0] == "Lee"


def test_link_permno_uses_link_valid_on_date_and_prefers_primary():
    ev = sample.load_gentry(gentry_frame())
    ccm = pd.DataFrame({"gvkey": ["001004"] * 3, "LINKPRIM": ["C", "P", "P"], "LIID": ["01"] * 3,
                        "LINKTYPE": ["LC", "LU", "LU"], "LPERMNO": [111, 222, 333], "LPERMCO": [1, 1, 1],
                        "LINKDT": ["2000-01-01", "2000-01-01", "2011-01-01"],
                        "LINKENDDT": ["E", "2010-12-31", "E"]})
    assert sample.link_permno(ev, ccm)["permno"].iloc[0] == 222


def test_first_departure_news_prefers_story_naming_the_ceo_and_rolls_after_close():
    ev = sample.load_gentry(gentry_frame()).assign(rp_entity_id="E1")
    rp = pd.DataFrame({
        "rp_entity_id": ["E1", "E1", "E1"],
        "timestamp_utc": pd.to_datetime(["2010-05-20 14:00", "2010-05-28 21:30", "2009-01-01 12:00"]),
        "type": ["executive-resignation", "executive-firing", "executive-firing"],
        "position_name": ["Chief Executive Officer"] * 3,
        "headline": ["A says its CEO will step down", "A ousts CEO Lee over conduct", "old story about Lee"],
        "event_text": ["", "", ""], "rp_story_id": ["s1", "s2", "s3"]})
    out = sample.choose_event_date(sample.first_departure_news(ev, rp))
    row = out.iloc[0]
    assert row["rp_n_stories"] == 2  # the 2009 story is outside the 400-day window
    assert row["rp_named_headline"] == "A ousts CEO Lee over conduct"
    # 21:30 UTC is 17:30 in New York, after the close, so the market reacts on 29 May
    assert row["event_date"] == pd.Timestamp("2010-05-29") and row["date_source"] == "ravenpack"
    assert row["gentry_gap_days"] == -3


def test_stage3_sql_builders():
    import importlib.util
    from pathlib import Path

    import pytest
    pytest.importorskip("wrds")
    spec = importlib.util.spec_from_file_location(
        "stage3", Path(__file__).resolve().parents[1] / "scripts" / "wrds_pull_stage3.py")
    st = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st)
    s = pd.DataFrame({"permno": pd.array([10, 10, 20, None], dtype="Int64"),
                      "event_date": pd.to_datetime(["2010-01-01", "2012-01-01", "2015-06-01", "2016-01-01"])})
    w = st.firm_windows(s)
    assert len(w) == 2 and str(w.iloc[0]["d1"]) == "2013-02-04"
    assert "(10, DATE '2008-05-11', DATE '2013-02-04')" in st.dsf_sql(w)
    sql = st.ravenpack_sql(2005, {"E1"})
    assert "rpa_djpr_equities_2005" in sql and "'E1'" in sql and "NOT IN ('executive-scandal'" in sql


def test_stage4_rdq_sql():
    import importlib.util
    from pathlib import Path

    import pytest
    pytest.importorskip("wrds")
    spec = importlib.util.spec_from_file_location(
        "stage4", Path(__file__).resolve().parents[1] / "scripts" / "wrds_pull_stage4.py")
    st = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st)
    sql = st.rdq_sql(["001004", "001004", "002000"])
    assert "comp.fundq" in sql and "gvkey IN ('001004', '002000')" in sql and "popsrc = 'D'" in sql


def test_stage4_headline_search_and_permnos():
    import importlib.util
    import re
    from pathlib import Path

    import pytest
    pytest.importorskip("wrds")
    spec = importlib.util.spec_from_file_location(
        "stage4b", Path(__file__).resolve().parents[1] / "scripts" / "wrds_pull_stage4.py")
    st = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st)
    assert "rpa_djpr_equities_2018" in st.headline_sql(2018) and "%" not in st.headline_sql(2018)
    hit = "Intel CEO Resigns After Probe Finds Past Relationship With Employee"
    miss = "Intel CEO Sees Strong Demand for Chips"
    both = lambda h: bool(re.search(st.PERSON, h, re.I) and re.search(st.SCANDAL, h, re.I))
    assert both(hit) and not both(miss)
    ccm = pd.DataFrame({"gvkey": ["000001", "000001", "000002"], "LINKTYPE": ["LC", "NU", "LU"],
                        "LINKPRIM": ["P", "P", "C"], "LPERMNO": [11, 12, 21],
                        "LINKENDDT": ["E", "E", "1995-01-01"]})
    assert st.execucomp_permnos(pd.DataFrame({"gvkey": ["000001", "000002"]}), ccm) == [11]
