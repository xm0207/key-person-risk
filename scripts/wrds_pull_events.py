"""Pull executive-level news events from RavenPack on WRDS, one year at a time.

Stage 2 of the data build. It takes every RavenPack event whose type is an executive or board
change, death, health problem or scandal, together with RavenPack's company mapping tables, which
link rp_entity_id to ticker, CUSIP and ISIN. The output defines the candidate CEO events.

Run it yourself in a terminal (password and Duo as before):

    ~/.venvs/key-person-risk/bin/python scripts/wrds_pull_events.py --user YOUR_WRDS_USERNAME

On a Mac, put `caffeinate -i` in front so the laptop does not go to sleep mid-run (a sleeping
laptop drops the connection: that is what stopped the first run at 2005).

The default edition, Dow Jones, has about 10-13 million rows a year: each yearly query scans one
table and should take about a minute, so all of 2000-2026 takes roughly 15-30 minutes. Years
already saved are skipped, so if it stops you can simply run it again. Use --edition full for the
much larger Full edition (about 1.7 billion rows; hours rather than minutes).

If the connection drops, the script rolls back, reconnects once (WRDS may send another Duo push)
and retries that year. If the retry fails too it stops, rather than marking every later year as
failed; run it again to carry on from where it stopped. TCP keepalives make a dead connection
show up within about two minutes instead of hanging for hours.

Output (gitignored), in data/wrds/ravenpack/:
    {edition}_exec_{year}.parquet   one file per year
    company_mappings.parquet, company_mappings_hist.parquet
    pull_log.csv                    rows and seconds per query
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import wrds
from keyperson.wrdsio import COLUMNS, CONNECT_ARGS, EVENT_TYPES, fetch

OUT = Path(__file__).resolve().parents[1] / "data" / "wrds" / "ravenpack"

EDITIONS = {"dj": ("ravenpack_dj", "rpa_djpr_equities"), "full": ("ravenpack_full", "rpa_full_equities")}

def year_sql(library: str, prefix: str, year: int) -> str:
    types = ", ".join(f"'{t}'" for t in EVENT_TYPES)
    return f"SELECT {', '.join(COLUMNS)} FROM {library}.{prefix}_{year} WHERE type IN ({types})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--user", required=True, help="your WRDS username")
    ap.add_argument("--edition", choices=sorted(EDITIONS), default="dj")
    ap.add_argument("--start", type=int, default=2000)
    ap.add_argument("--end", type=int, default=2026)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    library, prefix = EDITIONS[args.edition]
    log_path = OUT / "pull_log.csv"
    log = pd.read_csv(log_path).to_dict("records") if log_path.exists() else []

    db = wrds.Connection(wrds_username=args.user, wrds_connect_args=CONNECT_ARGS)
    try:
        for name, sql in [
            ("company_mappings", "SELECT * FROM ravenpack_common.wrds_rpa_company_mappings"),
            ("company_mappings_hist", "SELECT * FROM ravenpack_common.wrds_rpa_company_mappings_hist"),
        ]:
            path = OUT / f"{name}.parquet"
            if path.exists():
                print(f"{name}: already saved, skipped")
                continue
            t0 = time.time()
            df = fetch(db, sql, name)
            df.to_parquet(path, index=False)
            log.append({"item": name, "rows": len(df), "seconds": round(time.time() - t0, 1)})
            print(f"{name}: {len(df):,} rows")

        for year in range(args.start, args.end + 1):
            path = OUT / f"{args.edition}_exec_{year}.parquet"
            if path.exists():
                print(f"{year}: already saved, skipped")
                continue
            t0 = time.time()
            try:
                df = fetch(db, year_sql(library, prefix, year), str(year))
            except Exception as exc:  # stop: later years would hit the same broken connection
                log.append({"item": f"{args.edition}_{year}", "rows": None, "seconds": None,
                            "error": str(exc)[:200]})
                print(f"Stopped at {year}. Run the same command again to carry on from {year}.")
                break
            df.to_parquet(path, index=False)
            secs = round(time.time() - t0, 1)
            log.append({"item": f"{args.edition}_{year}", "rows": len(df), "seconds": secs})
            print(f"{year}: {len(df):,} rows in {secs}s")
            pd.DataFrame(log).to_csv(log_path, index=False)
    finally:
        pd.DataFrame(log).to_csv(log_path, index=False)
        db.close()
    print(f"Done. Files in {OUT}")


if __name__ == "__main__":
    main()
