"""Stage 3: pull CRSP prices and RavenPack company news for the firms in the event sample only.

Run scripts/build_event_sample.py first. Then, in a terminal (password and Duo as before):

    caffeinate -i ~/.venvs/key-person-risk/bin/python scripts/wrds_pull_stage3.py --user YOUR_WRDS_USERNAME

What it pulls, in this order (each part is saved as soon as it arrives; parts already saved are skipped,
so if it stops you can run it again):
  1. crsp.dsi max date, to record where legacy CRSP daily data end.
  2. crsp.dsf daily returns, prices, shares and volume, only for each sample firm and only from
     600 calendar days before its first event to 400 days after its last (chunks of 200 firms).
  3. crsp.dsedelist (delisting returns) and crsp.dsenames (share code, exchange, SIC) for those firms.
  4. ff.factors_daily (Fama-French daily factors and the risk-free rate).
  5. RavenPack Dow Jones company news for the sample firms, 2000-2026, in the groups that carry the
     channels a CEO might use (insider trades, buybacks and rights plans, dividends, legal and regulatory
     news, investor relations, board changes). About 1-2 minutes a year; --skip-ravenpack leaves it out.

Expect roughly 20-50 minutes in all. Output (gitignored) goes to data/wrds/stage3/.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import wrds

from keyperson.wrdsio import COLUMNS, CONNECT_ARGS, EVENT_TYPES, dsf_sql, fetch, firm_windows

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data/derived/event_sample.parquet"
OUT = ROOT / "data/wrds/stage3"
CHUNK = 200
CHANNEL_GROUPS = ["insider-trading", "equity-actions", "dividends", "legal", "regulatory",
                  "investor-relations", "labor-issues"]


def in_list(values) -> str:
    return ", ".join(str(int(v)) for v in values)


def ravenpack_sql(year: int, entities) -> str:
    ids = ", ".join(f"'{e}'" for e in sorted(entities))
    groups = ", ".join(f"'{g}'" for g in CHANNEL_GROUPS)
    skip = ", ".join(f"'{t}'" for t in EVENT_TYPES)  # already pulled by wrds_pull_events.py
    return (f"SELECT {', '.join(COLUMNS)} FROM ravenpack_dj.rpa_djpr_equities_{year} "
            f"WHERE rp_entity_id IN ({ids}) AND \"group\" IN ({groups}) AND type NOT IN ({skip})")


def save(db, sql: str, path: Path, log: list) -> None:
    if path.exists():
        print(f"{path.name}: already saved, skipped")
        return
    t0 = time.time()
    df = fetch(db, sql, path.stem)
    df.to_parquet(path, index=False)
    secs = round(time.time() - t0, 1)
    log.append({"item": path.stem, "rows": len(df), "seconds": secs})
    print(f"{path.stem}: {len(df):,} rows in {secs}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--user", required=True, help="your WRDS username")
    ap.add_argument("--skip-ravenpack", action="store_true")
    args = ap.parse_args()

    sample = pd.read_parquet(SAMPLE)
    windows = firm_windows(sample)
    permnos = windows["permno"].tolist()
    entities = set(sample["rp_entity_id"].dropna())
    print(f"{len(permnos)} firms, {len(entities)} RavenPack entities")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dsf").mkdir(exist_ok=True)
    (OUT / "ravenpack").mkdir(exist_ok=True)
    log_path = OUT / "pull_log.csv"
    log = pd.read_csv(log_path).to_dict("records") if log_path.exists() else []

    db = wrds.Connection(wrds_username=args.user, wrds_connect_args=CONNECT_ARGS)
    try:
        save(db, "SELECT max(date) AS last_date FROM crsp.dsi", OUT / "crsp_dsi_last_date.parquet", log)
        for i in range(0, len(windows), CHUNK):
            save(db, dsf_sql(windows.iloc[i:i + CHUNK]), OUT / "dsf" / f"dsf_{i // CHUNK:03d}.parquet", log)
            pd.DataFrame(log).to_csv(log_path, index=False)
        save(db, f"SELECT permno, dlstdt, dlstcd, dlret FROM crsp.dsedelist WHERE permno IN ({in_list(permnos)})",
             OUT / "dsedelist.parquet", log)
        save(db, ("SELECT permno, namedt, nameendt, ncusip, ticker, comnam, shrcd, exchcd, siccd "
                  f"FROM crsp.dsenames WHERE permno IN ({in_list(permnos)})"), OUT / "dsenames.parquet", log)
        save(db, "SELECT * FROM ff.factors_daily WHERE date >= '1998-01-01'", OUT / "ff_factors_daily.parquet", log)
        pd.DataFrame(log).to_csv(log_path, index=False)
        if not args.skip_ravenpack:
            for year in range(2000, 2027):
                save(db, ravenpack_sql(year, entities), OUT / "ravenpack" / f"dj_channels_{year}.parquet", log)
                pd.DataFrame(log).to_csv(log_path, index=False)
    except Exception as exc:
        print(f"Stopped: {type(exc).__name__}: {str(exc)[:200]}. Run the same command again to carry on.")
    finally:
        pd.DataFrame(log).to_csv(log_path, index=False)
        db.close()
    print(f"Done. Files in {OUT}")


if __name__ == "__main__":
    main()
