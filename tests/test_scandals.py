"""Candidate scandal stories: sitting-CEO surname match, former-CEO and company-name exclusions, episodes."""
import pandas as pd

from keyperson import scandals


def test_candidates_and_episodes():
    heads = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(["2018-06-21 12:00", "2018-07-01 12:00", "2019-09-01 12:00", "2018-06-22 12:00",
                                         "2018-06-23 12:00"]),
        "rp_story_id": ["a", "b", "c", "d", "e"], "rp_entity_id": ["E"] * 5, "entity_name": ["Acme Corp"] * 5,
        "headline": ["Acme CEO Leung Quits Over Relationship", "Leung probe widens", "Acme CEO Leung apologizes",
                     "Former Acme CEO Leung sued", "Acme CEO says sales fine"]})
    anncomp = pd.DataFrame({"gvkey": ["000001"] * 2, "year": [2018, 2019], "ceoann": ["CEO", "CEO"],
                            "execid": ["X", "X"], "co_per_rol": [7, 7], "exec_fullname": ["Ann Leung", "Ann Leung"]})
    st = scandals.candidate_stories(heads, pd.DataFrame({"rp_entity_id": ["E"], "gvkey": ["000001"]}),
                                    scandals.sitting_ceos(anncomp))
    assert list(st["rp_story_id"]) == ["a", "b", "c"]  # 'former' story and the story without the name drop out
    ep = scandals.build_episodes(st)
    assert len(ep) == 2 and ep["stories_30d"].tolist() == [2, 1]  # c comes more than 365 days after b
    assert ep["keyword_class"].tolist() == ["personal", "personal"]


def test_company_name_surnames_are_dropped():
    heads = pd.DataFrame({"timestamp_utc": pd.to_datetime(["2005-01-01"]), "rp_story_id": ["a"], "rp_entity_id": ["E"],
                          "entity_name": ["Ford Motor Co."], "headline": ["Ford CEO Ford faces probe"]})
    anncomp = pd.DataFrame({"gvkey": ["000002"], "year": [2005], "ceoann": ["CEO"], "execid": ["F"],
                            "co_per_rol": [1], "exec_fullname": ["William Clay Ford"]})
    st = scandals.candidate_stories(heads, pd.DataFrame({"rp_entity_id": ["E"], "gvkey": ["000002"]}),
                                    scandals.sitting_ceos(anncomp))
    assert st.empty


def test_classify_disclosure_and_strict_words():
    assert scandals.classify_disclosure(pd.Timestamp("2024-09-01"), pd.Timestamp("2024-09-11")) == "two-stage"
    assert scandals.classify_disclosure(pd.Timestamp("2019-11-03"), pd.Timestamp("2019-11-04")) == "one-stage"
    assert scandals.classify_disclosure(pd.Timestamp("2019-02-07"), pd.NaT) == "survived"
    s = pd.DataFrame({"headline": ["Acme Semiconductor CEO Pleased With Sales", "Acme CEO Under Investigation"]})
    assert scandals.strict_stories(s)["headline"].tolist() == ["Acme CEO Under Investigation"]


def test_entity_with_two_gvkeys_and_stable_ids():
    heads = pd.DataFrame({"timestamp_utc": pd.to_datetime(["2021-11-16 15:59"]), "rp_story_id": ["k"], "rp_entity_id": ["A"],
                          "entity_name": ["Activision Blizzard Inc."], "headline": ["CEO Kotick Knew About Misconduct"]})
    anncomp = pd.DataFrame({"gvkey": ["180405"], "year": [2021], "ceoann": ["CEO"], "execid": ["K"],
                            "co_per_rol": [9], "exec_fullname": ["Robert A. Kotick"]})
    em = pd.DataFrame({"rp_entity_id": ["A", "A"], "gvkey": ["001111", "180405"]})  # old and new gvkey
    st = scandals.candidate_stories(heads, em, scandals.sitting_ceos(anncomp))
    assert st["gvkey"].tolist() == ["180405"]
    ep = scandals.build_episodes(st)
    prev = pd.DataFrame({"candidate_id": ["C0007"], "gvkey": [1], "execid": ["Z"], "first_story_id": ["z"], "first_ts": ["2020-01-01"]})
    assert scandals.stable_ids(ep, prev)["candidate_id"].tolist() == ["C0008"]
    prev = pd.DataFrame({"candidate_id": ["C0003"], "gvkey": ["180405"], "execid": ["K"], "first_story_id": ["j"], "first_ts": ["2021-11-16 15:59"]})
    assert scandals.stable_ids(ep, prev)["candidate_id"].tolist() == ["C0003"]
