"""Prediction-market prices on Jensen Huang's position at Nvidia, October-November 2026, from the public
Kalshi and Polymarket APIs (no account needed).

    ~/.venvs/key-person-risk/bin/python scripts/prediction_markets.py discover
        lists Kalshi series and Polymarket events whose titles mention Nvidia or Huang
    ~/.venvs/key-person-risk/bin/python scripts/prediction_markets.py fetch
        downloads price histories for the markets in MARKETS to data/prediction_markets/ (not committed)

Raw prices stay in data/; only prices at the times listed in TIMES are written to results/.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/prediction_markets"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
HEADERS = {"User-Agent": "keyperson-research-script/0.1", "Accept": "application/json"}
WORDS = ("nvidia", "huang")


def get(url: str, **params) -> dict:
    for wait in (2, 10, 30, 0):
        r = requests.get(url, params=params, headers=HEADERS, timeout=60)
        if r.status_code != 429 or not wait:
            break
        time.sleep(wait)
    r.raise_for_status()
    return r.json()


def discover() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    series = get(f"{KALSHI}/series").get("series", [])
    hits = [s for s in series if any(w in (s.get("title", "") + " " + s.get("ticker", "")).lower() for w in WORDS)]
    print(f"Kalshi: {len(series)} series, {len(hits)} mention Nvidia or Huang")
    found = {"kalshi": [], "polymarket": []}
    for s in hits:
        print(f"  series {s['ticker']}: {s.get('title')}")
        markets, cursor = [], None
        while True:
            page = get(f"{KALSHI}/markets", series_ticker=s["ticker"], limit=1000, **({"cursor": cursor} if cursor else {}))
            markets += page.get("markets", [])
            cursor = page.get("cursor")
            if not cursor:
                break
        for m in markets:
            print(f"    {m['ticker']} | {m.get('title')} | {m.get('yes_sub_title') or m.get('subtitle')} | "
                  f"status {m.get('status')} | open {m.get('open_time')} | last {m.get('last_price')}")
            found["kalshi"].append({"series": s["ticker"], **{k: m.get(k) for k in
                                    ["ticker", "event_ticker", "title", "yes_sub_title", "subtitle", "status", "open_time", "close_time"]}})
    for q in ["Jensen Huang", "Nvidia CEO"]:
        res = get(f"{GAMMA}/public-search", q=q)
        for e in res.get("events", []) or []:
            print(f"Polymarket event {e.get('slug')} | {e.get('title')} | start {e.get('startDate')}")
            for m in e.get("markets", []) or []:
                print(f"    {m.get('slug')} | {m.get('question')} | tokens {m.get('clobTokenIds')}")
                found["polymarket"].append({"event": e.get("slug"), "event_title": e.get("title"), "slug": m.get("slug"),
                                            "question": m.get("question"), "tokens": m.get("clobTokenIds")})
    (OUT / "markets_found.json").write_text(json.dumps(found, indent=1))


ET = "America/New_York"
KALSHI_MARKETS = {  # series, ticker, description
    "exit_by_2029": ("KXCEONVDA", "KXCEONVDA-01JAN33-01JAN29", "Huang leaves as CEO before 1 Jan 2029"),
    "exit_by_2027": ("KXCEONVDA", "KXCEONVDA-01JAN33-01JAN27", "Huang leaves as CEO before 1 Jan 2027"),
    "exit_by_dec1": ("KXCEONVDA", "KXCEONVDA-01JAN33-01DEC26", "Huang leaves as CEO before 1 Dec 2026"),
    "next_ceo_huang": ("KXNVDACEO", "KXNVDACEO-29JAN-JENS", "Huang named the next permanent CEO (before 2029)"),
}
POLY_EVENTS = ["jensen-huang-rejoins-nvidia-byptptpt", "jensen-huang-named-nvidia-ceo-and-nvidia-market-cap-above-4t-end-of-2026",
               "new-nvidia-ceo-named-by-december-31-2026", "next-nvidia-ceo-internal-or-external"]
MINUTE_WINDOWS = [("2026-11-08 12:00", "2026-11-09 10:30"), ("2026-11-11 15:30", "2026-11-12 11:00"),
                  ("2026-11-16 12:00", "2026-11-16 14:00"), ("2026-11-17 10:30", "2026-11-17 11:30"),
                  ("2026-11-17 15:30", "2026-11-17 20:00")]


def ts(s: str) -> int:
    return int(pd.Timestamp(s, tz=ET).timestamp())


def kalshi_candles(series: str, ticker: str, start: str, end: str, period: int) -> pd.DataFrame:
    j = get(f"{KALSHI}/series/{series}/markets/{ticker}/candlesticks", start_ts=ts(start), end_ts=ts(end),
            period_interval=period)
    rows = []
    for c in j.get("candlesticks", []):
        price, bid, ask = c.get("price") or {}, c.get("yes_bid") or {}, c.get("yes_ask") or {}
        num = lambda d, k: float(d[k]) if d.get(k) not in (None, "") else None
        rows.append({"ts": pd.to_datetime(c["end_period_ts"], unit="s", utc=True).tz_convert(ET),
                     "last": num(price, "close_dollars"), "bid": num(bid, "close_dollars"), "ask": num(ask, "close_dollars"),
                     "volume": num(c, "volume_fp"), "open_interest": num(c, "open_interest_fp")})
    return pd.DataFrame(rows)


def poly_history(token: str, start: str, end: str, fidelity: int) -> pd.DataFrame:
    j = get(f"{CLOB}/prices-history", market=token, startTs=ts(start), endTs=ts(end), fidelity=fidelity)
    h = j.get("history", [])
    return pd.DataFrame({"ts": pd.to_datetime([x["t"] for x in h], unit="s", utc=True).tz_convert(ET),
                         "p": [x["p"] for x in h]}) if h else pd.DataFrame(columns=["ts", "p"])


def yahoo_5m(symbol: str) -> pd.DataFrame:
    j = get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}", interval="5m", range="10d",
            includePrePost="true")["chart"]["result"][0]
    q = j["indicators"]["quote"][0]
    return pd.DataFrame({"ts": pd.to_datetime(j["timestamp"], unit="s", utc=True).tz_convert(ET), "close": q["close"]})


def fetch() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for key, (series, ticker, _) in KALSHI_MARKETS.items():
        frames = [kalshi_candles(series, ticker, "2026-10-29 00:00", "2026-11-18 23:59", 60).assign(res="1h")]
        for a, b in MINUTE_WINDOWS:
            frames.append(kalshi_candles(series, ticker, a, b, 1).assign(res="1m"))
        df = pd.concat(frames)
        df.to_parquet(OUT / f"kalshi_{key}.parquet")
        print(f"kalshi {key}: {len(df)} candles, {df['ts'].min()} to {df['ts'].max()}")
    for slug in POLY_EVENTS:
        ev = get(f"{GAMMA}/events", slug=slug)
        ev = ev[0] if isinstance(ev, list) and ev else ev
        for m in ev.get("markets", []):
            tokens = json.loads(m["clobTokenIds"]) if isinstance(m.get("clobTokenIds"), str) else m.get("clobTokenIds")
            frames = [poly_history(tokens[0], "2026-11-12 00:00", "2026-11-18 23:59", 60).assign(res="1h")]
            for a, b in MINUTE_WINDOWS[2:]:
                frames.append(poly_history(tokens[0], a, b, 1).assign(res="1m"))
            df = pd.concat(frames).assign(question=m.get("question"), event=slug)
            name = (m.get("slug") or m.get("question"))[:80].replace("/", "_")
            df.to_parquet(OUT / f"poly_{name}.parquet")
            print(f"polymarket {m.get('question')[:70]}: {len(df)} points")
    for sym in ["NVDA", "SPY"]:
        yahoo_5m(sym).to_parquet(OUT / f"yahoo5m_{sym}.parquet")
        print(f"yahoo 5m {sym} saved")


KEY_TIMES = ["2026-11-06 16:00", "2026-11-08 18:00", "2026-11-10 16:00", "2026-11-11 15:59", "2026-11-11 17:30",
             "2026-11-11 19:59", "2026-11-12 08:50", "2026-11-16 12:35", "2026-11-16 13:30", "2026-11-17 10:55",
             "2026-11-17 11:05", "2026-11-17 16:55", "2026-11-17 17:15", "2026-11-18 09:00"]
NVDA_BETA_SPY = 2.07  # Nvidia's daily beta on SPY in the paper's benchmark (nvidia_attribution.py, model A)
# Odds moves and Nvidia's move over the same minutes: (label, start, end)
CO_MOVES = [("16 Nov: report that the board is not considering a return", "2026-11-16 12:35", "2026-11-16 13:30"),
            ("17 Nov: offer to return", "2026-11-17 10:55", "2026-11-17 11:05"),
            ("17 Nov: chair rules out a return on the call", "2026-11-17 16:55", "2026-11-17 17:15")]


def analyze() -> None:
    def kal(name):
        d = pd.read_parquet(OUT / f"kalshi_{name}.parquet").sort_values("ts").drop_duplicates("ts", keep="last")
        return d.assign(mid=(d["bid"] + d["ask"]) / 2).dropna(subset=["mid"])

    def at(d, t, col="mid"):
        x = d[d["ts"] <= pd.Timestamp(t, tz=ET)]
        return x[col].iloc[-1] if len(x) else float("nan")

    rows = []
    for t in KEY_TIMES:
        row = {"time_et": t}
        for name in ["exit_by_2029", "exit_by_2027", "exit_by_dec1", "next_ceo_huang"]:
            settled = name.startswith("exit") and pd.Timestamp(t, tz=ET) > pd.Timestamp("2026-11-12 09:30", tz=ET)
            row[name] = float("nan") if settled else at(kal(name), t)  # the exit markets settled on 12 November
        rows.append(row)
    pd.DataFrame(rows).to_csv(ROOT / "results/prediction_market_probabilities.csv", index=False)

    nv = pd.read_parquet(OUT / "yahoo5m_NVDA.parquet").dropna().set_index("ts")["close"]
    sp = pd.read_parquet(OUT / "yahoo5m_SPY.parquet").dropna().set_index("ts")["close"]
    last = lambda s, t: s[s.index <= pd.Timestamp(t, tz=ET)].iloc[-1]
    q = kal("next_ceo_huang")
    co = []
    for label, a, b in CO_MOVES:
        dn = np.log(last(nv, b) / last(nv, a)) - NVDA_BETA_SPY * np.log(last(sp, b) / last(sp, a))
        co.append({"event": label, "start_et": a, "end_et": b, "odds_start": at(q, a), "odds_end": at(q, b),
                   "nvidia_abnormal": dn})
    co = pd.DataFrame(co)
    dq = co["odds_end"] - co["odds_start"]
    slope = float((dq * co["nvidia_abnormal"]).sum() / (dq ** 2).sum())
    co.loc[len(co)] = {"event": "slope: Nvidia abnormal return per unit of odds (value of a certain return)",
                       "nvidia_abnormal": slope}
    co.to_csv(ROOT / "results/prediction_market_comoves.csv", index=False)

    joint = {}
    for f in OUT.glob("poly_*.parquet"):
        d = pd.read_parquet(f).dropna(subset=["p"]).sort_values("ts")
        if "market cap" in d["question"].iloc[0]:
            joint[d["question"].iloc[0]] = (d["p"].iloc[-1], d["ts"].iloc[-1])
    get_p = lambda named, above: next(v[0] for k, v in joint.items()
                                      if (("not be named" not in k) == named) and (("above $4T" in k) == above))
    hn_a, hn_b, nn_a, nn_b = get_p(True, True), get_p(True, False), get_p(False, True), get_p(False, False)
    cond = pd.DataFrame([{"as_of_et": max(v[1] for v in joint.values()),
                          "p_named": hn_a + hn_b, "p_above_4T_if_named": hn_a / (hn_a + hn_b),
                          "p_above_4T_if_not_named": nn_a / (nn_a + nn_b), "sum_of_prices": hn_a + hn_b + nn_a + nn_b}])
    cond.to_csv(ROOT / "results/prediction_market_conditional.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(co.round(4).to_string(index=False))
    print(cond.round(3).to_string(index=False))


def raw() -> None:
    j = get(f"{KALSHI}/series/KXCEONVDA/markets/KXCEONVDA-01JAN33-01DEC26/candlesticks",
            start_ts=ts("2026-11-11 19:00"), end_ts=ts("2026-11-11 20:00"), period_interval=60)
    print(json.dumps(j.get("candlesticks", [])[:1], indent=1)[:2500])


if __name__ == "__main__":
    {"discover": discover, "fetch": fetch, "analyze": analyze, "raw": raw}[sys.argv[1] if len(sys.argv) > 1 else "discover"]()
