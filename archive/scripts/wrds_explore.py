"""Map what the WRDS PostgreSQL server offers for the key-person study, before any large pull.

Run it yourself in a terminal. The wrds client asks for your password and sends a Duo push; this
script never sees or stores credentials (answer "n" if asked to create a .pgpass file, unless you
want one). It only reads metadata and runs a few GROUP BY counts, so it is quick and light.

    ~/.venvs/key-person-risk/bin/python scripts/wrds_explore.py --user YOUR_WRDS_USERNAME

Output (gitignored, under data/wrds/explore/):
    libraries.csv            every library your account can see
    tables.csv               tables in the CRSP, Compustat, ExecuComp and RavenPack libraries
    ravenpack_columns.csv    columns of every RavenPack table
    ravenpack_taxonomy.csv   event-category counts in the latest yearly RavenPack equities table,
                             restricted to categories that look executive- or scandal-related
    ravenpack_entity_types.csv  entity types in RavenPack's entity mapping table, if there is one
    log.txt                  what ran, what failed and why
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import wrds

OUT = Path(__file__).resolve().parents[2] / "data" / "wrds" / "explore"

# Libraries worth listing in full.
WANTED = re.compile(r"raven|^rpa|^crsp|^comp$|^comp_|execcomp|^ff$")
RAVEN = re.compile(r"raven|^rpa")

# Category fragments that suggest an executive-level or misconduct event.
CATEGORY_HINTS = [
    "executive", "ceo", "chairman", "founder", "management", "scandal", "harass", "misconduct",
    "sexual", "fraud", "bribery", "corruption", "insider", "investigation", "probe", "allegation",
    "lawsuit", "legal", "resign", "dismiss", "fired", "ousted", "departure", "succession",
    "death", "health", "illness", "leave", "apolog", "conduct", "ethic", "spy", "espionage",
]


class Explorer:
    def __init__(self, user: str):
        self.db = wrds.Connection(wrds_username=user)
        self.log: list[str] = []

    def note(self, msg: str) -> None:
        print(msg)
        self.log.append(msg)

    def safe(self, label: str, fn):
        try:
            return fn()
        except Exception as exc:  # keep going: one unavailable table must not stop the survey
            self.note(f"FAILED {label}: {type(exc).__name__}: {str(exc)[:300]}")
            return None

    def run(self) -> None:
        OUT.mkdir(parents=True, exist_ok=True)

        libs = sorted(self.db.list_libraries())
        pd.Series(libs, name="library").to_csv(OUT / "libraries.csv", index=False)
        self.note(f"{len(libs)} libraries visible")

        rows = []
        for lib in (l for l in libs if WANTED.search(l)):
            for t in self.safe(f"list_tables {lib}", lambda: self.db.list_tables(library=lib)) or []:
                rows.append({"library": lib, "table": t})
        tables = pd.DataFrame(rows, columns=["library", "table"])
        tables.to_csv(OUT / "tables.csv", index=False)
        self.note(f"{len(tables)} tables listed in {tables.library.nunique()} libraries")

        rp = tables[tables.library.str.contains(RAVEN)]
        cols = []
        for lib, tab in rp.itertuples(index=False):
            d = self.safe(f"describe {lib}.{tab}", lambda: self.db.describe_table(lib, tab))
            if d is not None:
                n = self.safe(f"row count {lib}.{tab}", lambda: self.db.get_row_count(lib, tab))
                d = d.assign(library=lib, table=tab, approx_rows=n)
                cols.append(d)
        if not cols:
            self.note("No RavenPack tables could be described")
            self.finish()
            return
        columns = pd.concat(cols, ignore_index=True)
        columns.to_csv(OUT / "ravenpack_columns.csv", index=False)

        self.taxonomy(columns)
        self.entity_types(columns)
        self.finish()

    def taxonomy(self, columns: pd.DataFrame) -> None:
        """Category counts in the latest yearly equities table that has a category column."""
        has_cat = columns[columns["name"].str.lower() == "category"][["library", "table"]].drop_duplicates()
        yearly = has_cat[has_cat["table"].str.contains(r"equit", case=False)
                         & has_cat["table"].str.contains(r"(19|20)\d\d")].copy()
        if yearly.empty:
            yearly = has_cat.copy()
        if yearly.empty:
            self.note("No RavenPack table with a 'category' column")
            return
        yearly["year"] = yearly["table"].str.extract(r"((?:19|20)\d\d)").astype(float)
        lib, tab = yearly.sort_values("year", na_position="first").iloc[-1][["library", "table"]]
        names = set(columns.loc[(columns.library == lib) & (columns.table == tab), "name"].str.lower())
        keys = [c for c in ["group", "type", "sub_type", "property", "category"] if c in names]
        select = ", ".join(f'"{c}"' for c in keys)
        like = " OR ".join(f"lower(category) LIKE '%%{h}%%'" for h in CATEGORY_HINTS)
        sql = (f"SELECT {select}, count(*) AS n FROM {lib}.{tab} "
               f"WHERE {like} GROUP BY {select} ORDER BY n DESC")
        self.note(f"Taxonomy from {lib}.{tab}")
        df = self.safe(f"taxonomy {lib}.{tab}", lambda: self.db.raw_sql(sql))
        if df is not None:
            df.insert(0, "source_table", f"{lib}.{tab}")
            df.to_csv(OUT / "ravenpack_taxonomy.csv", index=False)
            self.note(f"{len(df)} executive- or misconduct-related categories found")

    def entity_types(self, columns: pd.DataFrame) -> None:
        """Does RavenPack track people (for example CEOs) as entities?"""
        cand = columns[columns["name"].str.lower() == "entity_type"][["library", "table"]].drop_duplicates()
        cand = cand[cand["table"].str.contains("map|entit", case=False)]
        if cand.empty:
            self.note("No entity mapping table with an entity_type column")
            return
        lib, tab = cand.iloc[0]
        sql = f"SELECT entity_type, count(*) AS n FROM {lib}.{tab} GROUP BY entity_type ORDER BY n DESC"
        df = self.safe(f"entity types {lib}.{tab}", lambda: self.db.raw_sql(sql))
        if df is not None:
            df.insert(0, "source_table", f"{lib}.{tab}")
            df.to_csv(OUT / "ravenpack_entity_types.csv", index=False)

    def finish(self) -> None:
        (OUT / "log.txt").write_text("\n".join(self.log) + "\n")
        self.db.close()
        print(f"Done. Files in {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--user", required=True, help="your WRDS username")
    Explorer(ap.parse_args().user).run()


if __name__ == "__main__":
    main()
