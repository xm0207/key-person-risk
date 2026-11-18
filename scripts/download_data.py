"""Build data/keyperson.duckdb: daily adjusted closes from Yahoo Finance for the tickers the events need.

Prices come through mega7's `YahooClient` and are stored with mega7's `PriceStore` (table
observations(dataset, series, date, value), dataset 'price'). Yahoo data are for personal use: the
database lives in data/, which is not committed, and nothing in results/ is a price or return series.

Every series is cut at `--end` (default: the project's data cut-off, 17 November 2026; 13 November until 18 November 2026). On 16 November
2026 a fresh download agreed with the mega7-decomposition database (downloaded 11 November) to within
2e-6 in every daily return up to 10 November 2026.

The engine does not depend on this source: any DataFrame of daily returns (for example from CRSP) works.
"""

from __future__ import annotations

import argparse

from mega7 import PriceStore, YahooClient

from keyperson.config import ALL_TICKERS, DATA_END


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default="data/keyperson.duckdb")
    parser.add_argument("--end", default=DATA_END)
    parser.add_argument("--tickers", nargs="*", default=ALL_TICKERS)
    args = parser.parse_args()

    store = PriceStore(args.db)
    yahoo = YahooClient()
    for ticker in args.tickers:
        frame = yahoo.fetch(ticker, end=args.end)
        store.upsert("price", frame)
        print(f"{ticker:<6} {len(frame):>6} closes  {frame['date'].min().date()} to {frame['date'].max().date()}")
    print(store.coverage().to_string(index=False))
    store.close()


if __name__ == "__main__":
    main()
