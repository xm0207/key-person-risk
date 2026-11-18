"""Episode construction: the 365-day gap rule, the after-close date roll and the CEO outcome."""
import pandas as pd

from keyperson import episodes


def stories():
    return pd.DataFrame({
        "timestamp_utc": ["2005-03-01 14:00", "2005-03-10 22:00", "2006-06-01 21:30", "2005-01-05 10:00"],
        "rp_entity_id": ["A", "A", "A", "B"], "entity_name": ["Acme", "Acme", "Acme", "Bco"],
        "type": ["fraud", "regulatory-investigation", "fraud", "lawsuit"],
        "event_sentiment_score": [-0.5, -0.4, -0.6, -0.9], "news_type": ["FULL-ARTICLE"] * 4,
        "relevance": [100, 100, 95, 100], "headline": ["h1", "h2", "h3", "h4"], "rp_story_id": ["1", "2", "3", "4"]})


def test_gap_rule_and_after_close_roll():
    ep = episodes.build_episodes(episodes.qualifying_stories(stories()))
    assert list(ep["headline"]) == ["h1", "h3"]  # h2 is within a year of h1; h4 is not a misconduct type
    assert ep["stories_30d"].tolist() == [2, 1]
    # 21:30 UTC on 1 June 2006 is 17:30 in New York, after the close
    assert ep["event_date"].iloc[1] == pd.Timestamp("2006-06-02")


def test_ceo_outcome_within_a_year():
    ep = episodes.build_episodes(episodes.qualifying_stories(stories()))
    anncomp = pd.DataFrame({"gvkey": ["000001"] * 2, "year": [2005, 2006], "ceoann": ["CEO", "CEO"],
                            "co_per_rol": [7, 8], "execid": ["X", "Y"]})
    deps = pd.DataFrame({"co_per_rol": [7], "ref_date": [pd.Timestamp("2005-12-01")], "departure_code": [4],
                         "reason": ["dismissed_misconduct"]})
    out = episodes.attach_ceo_and_outcome(ep, pd.DataFrame({"rp_entity_id": ["A"], "gvkey": ["000001"]}), anncomp, deps)
    assert out["ceo_left_within_year"].tolist() == [1.0, 0.0]
    assert out["exit_reason"].iloc[0] == "dismissed_misconduct"
