"""The committed results/ and archive/results/ may hold summary statistics only, never a daily or weekly series
(Yahoo and WRDS data may not be redistributed). This checks every CSV in both with mega7's rule."""

from pathlib import Path

import pandas as pd
import pytest

from mega7.results import looks_like_high_frequency_series

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(ROOT.glob("results/*.csv")) + sorted(ROOT.glob("archive/results/*.csv"))


@pytest.mark.parametrize("path", FILES, ids=[str(p.relative_to(ROOT)) for p in FILES])
def test_no_daily_series_in_results(path):
    frame = pd.read_csv(path)
    for column in frame.columns:
        if "date" in column:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    assert not looks_like_high_frequency_series(frame)


def test_event_list_has_sources_for_every_included_event():
    events = pd.read_csv(Path(__file__).resolve().parents[1] / "events" / "megacap_events.csv")
    included = events[events["status"].str.startswith("included") | events["status"].str.startswith("excluded")]
    assert included["primary_source_url"].str.startswith("https://").all()
    assert included["timing"].isin(["before_open", "during_hours", "after_close", "non_trading_day"]).all()
    assert included["day0"].notna().all()
