"""Download RavenPack's event taxonomy and a small sample of executive-level events.

These queries are quick. The taxonomy is a 6,688-row table; the entity-type count runs on a
289k-row table; the sample stops after 500 matching rows. Run it yourself in a terminal (password
and Duo as before):

    ~/.venvs/key-person-risk/bin/python scripts/wrds_ravenpack_taxonomy.py --user YOUR_WRDS_USERNAME

Output (gitignored, under data/wrds/explore/):
    rpa_taxonomy.csv            the full RavenPack event taxonomy (topic, group, type, ..., category, description)
    rpa_entity_types.csv        how many entities RavenPack tracks by type (companies, people, ...)
    rpa_sample_executive.csv    500 recent US events whose category mentions executives, with headlines
"""
from __future__ import annotations

import argparse
from pathlib import Path

import wrds

OUT = Path(__file__).resolve().parents[2] / "data" / "wrds" / "explore"

SAMPLE_SQL = """
SELECT rpa_date_utc, rpa_time_utc, entity_name, entity_type, country_code, relevance,
       event_sentiment_score, event_relevance, event_similarity_days, "group", type, sub_type,
       property, category, rp_position_id, position_name, related_entity, relationship,
       event_text, news_type, source_name, headline
FROM ravenpack_full.rpa_full_equities_2026
WHERE country_code = 'US'
  AND (category LIKE 'executive%%' OR "group" = 'labor-issues' OR "group" = 'legal')
LIMIT 500
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--user", required=True, help="your WRDS username")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    db = wrds.Connection(wrds_username=args.user)
    try:
        jobs = {
            "rpa_taxonomy.csv": "SELECT * FROM ravenpack_common.rpa_taxonomy",
            "rpa_entity_types.csv": ("SELECT entity_type, count(*) AS n FROM ravenpack_common.wrds_rpa_entity_mappings "
                                     "GROUP BY entity_type ORDER BY n DESC"),
            "rpa_sample_executive.csv": SAMPLE_SQL,
        }
        for name, sql in jobs.items():
            try:
                df = db.raw_sql(sql)
                df.to_csv(OUT / name, index=False)
                print(f"{name}: {len(df)} rows")
            except Exception as exc:  # report and continue
                print(f"FAILED {name}: {type(exc).__name__}: {str(exc)[:300]}")
    finally:
        db.close()
    print(f"Done. Files in {OUT}")


if __name__ == "__main__":
    main()
