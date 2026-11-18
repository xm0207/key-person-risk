"""Stage 4: data for CEO personal-scandal episodes (the Nvidia kind), whether or not the CEO survived.

    caffeinate -i ~/.venvs/key-person-risk/bin/python scripts/wrds_pull_stage4.py --user YOUR_WRDS_USERNAME

Four parts, each saved as it arrives; run the same command again to resume after a stop.
  1. Compustat quarterly earnings announcement dates (`rdq`) for every ExecuComp firm: forced exits are
     often announced with bad earnings, which must be separated from the CEO news.
  2. RavenPack Dow Jones stories, 2000-2026, about US companies, whose headline mentions a CEO, chief
     executive, chairman or founder AND a scandal word (affair, harassment, misconduct, probe, lawsuit,
     indictment, apology, leave of absence, ...). Matching the sitting CEO's surname and coding the
     episodes happens locally afterwards. About 1-3 minutes a year.
  3. CRSP daily returns, 1999-2024, for every PERMNO linked to an ExecuComp firm: the full panel, so that
     any episode has returns and industry benchmarks can exclude the event firm. About 20-40 minutes.
  4. CRSP delisting returns and names (share code, SIC) for the same PERMNOs.

Expect 1-1.5 hours in all. Output (gitignored): data/wrds/stage4/.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import wrds

from keyperson.wrdsio import COLUMNS, CONNECT_ARGS, fetch

ROOT = Path(__file__).resolve().parents[1]
WRDS = ROOT / "data/wrds"
OUT = WRDS / "stage4"
CHUNK = 150

PERSON = r"(ceo|chief executive|chairman|chairwoman|founder)"
SCANDAL = (r"(affair|relationship|sexual|harass|misconduct|inappropriate|allegation|alleged|accus|scandal|probe|"
           r"investigat|inquiry|ethic|expense|lawsuit|sued|subpoena|indict|charged|arrest|guilty|plea|fraud|"
           r"insider trading|misled|lied|conduct|leave of absence|apolog|controvers|outrage|backlash|tweet)")


def rdq_sql(gvkeys) -> str:
    keys = ", ".join(f"'{g}'" for g in sorted(set(gvkeys)))
    return ("SELECT gvkey, datadate, fyearq, fqtr, rdq FROM comp.fundq "
            "WHERE indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C' "
            f"AND datadate >= '1998-01-01' AND gvkey IN ({keys})")


def headline_sql(year: int) -> str:
    return (f"SELECT {', '.join(COLUMNS)} FROM ravenpack_dj.rpa_djpr_equities_{year} "
            f"WHERE country_code = 'US' AND entity_type = 'COMP' "
            f"AND headline ~* '{PERSON}' AND headline ~* '{SCANDAL}'")


def execucomp_permnos(anncomp: pd.DataFrame, ccm: pd.DataFrame) -> list[int]:
    gv = set(anncomp["gvkey"].astype(str).str.zfill(6))
    c = ccm.rename(columns=str.lower)
    c = c[c["linktype"].isin(["LC", "LU"]) & c["linkprim"].isin(["P", "C"])]
    c = c[c["gvkey"].astype(str).str.zfill(6).isin(gv)]
    end = pd.to_datetime(c["linkenddt"].replace("E", None), errors="coerce").fillna(pd.Timestamp("2099-12-31"))
    c = c[end >= pd.Timestamp("1999-01-01")]
    return sorted(c["lpermno"].dropna().astype(int).unique().tolist())


def dsf_sql(permnos) -> str:
    ids = ", ".join(str(int(p)) for p in permnos)
    return ("SELECT permno, date, ret, retx, prc, shrout, vol FROM crsp.dsf "
            f"WHERE permno IN ({ids}) AND date BETWEEN '1999-01-01' AND '2024-12-31'")


def save(db, sql: str, path: Path, log: list) -> None:
    if path.exists():
        print(f"{path.name}: already saved, skipped")
        return
    t0 = time.time()
    df = fetch(db, sql, path.stem)
    df.to_parquet(path, index=False)
    log.append({"item": path.stem, "rows": len(df), "seconds": round(time.time() - t0, 1)})
    print(f"{path.stem}: {len(df):,} rows in {log[-1]['seconds']}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--user", required=True, help="your WRDS username")
    args = ap.parse_args()

    anncomp = pd.read_csv(WRDS / "execucomp_anncomp.csv", usecols=["gvkey"], dtype={"gvkey": str})
    permnos = execucomp_permnos(anncomp, pd.read_csv(WRDS / "ccm_link_table.csv", dtype={"gvkey": str}))
    print(f"{anncomp['gvkey'].nunique():,} ExecuComp firms, {len(permnos):,} linked PERMNOs")
    for sub in ["", "ravenpack", "dsf"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    log_path = OUT / "pull_log.csv"
    log = pd.read_csv(log_path).to_dict("records") if log_path.exists() else []

    db = wrds.Connection(wrds_username=args.user, wrds_connect_args=CONNECT_ARGS)
    try:
        save(db, rdq_sql(anncomp["gvkey"]), OUT / "fundq_rdq.parquet", log)
        for year in range(2000, 2027):
            save(db, headline_sql(year), OUT / "ravenpack" / f"dj_ceo_scandal_{year}.parquet", log)
            pd.DataFrame(log).to_csv(log_path, index=False)
        for i in range(0, len(permnos), CHUNK):
            save(db, dsf_sql(permnos[i:i + CHUNK]), OUT / "dsf" / f"dsf_{i // CHUNK:03d}.parquet", log)
            pd.DataFrame(log).to_csv(log_path, index=False)
        ids = ", ".join(str(p) for p in permnos)
        save(db, f"SELECT permno, dlstdt, dlstcd, dlret FROM crsp.dsedelist WHERE permno IN ({ids})",
             OUT / "dsedelist.parquet", log)
        save(db, ("SELECT permno, namedt, nameendt, ncusip, ticker, comnam, shrcd, exchcd, siccd "
                  f"FROM crsp.dsenames WHERE permno IN ({ids})"), OUT / "dsenames.parquet", log)
    except Exception as exc:
        print(f"Stopped: {type(exc).__name__}: {str(exc)[:200]}. Run the same command again to carry on.")
    finally:
        pd.DataFrame(log).to_csv(log_path, index=False)
        db.close()
    print(f"Done. Files in {OUT}")


if __name__ == "__main__":
    main()
