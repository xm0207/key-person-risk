"""Check every candidate reference against Crossref and write paper/references_verification.csv and paper/refs.bib.

For each candidate (first author, year, title, journal as remembered), the script queries
https://api.crossref.org/works?query.bibliographic=... and compares the best match with the candidate:
the title similarity, the year and the first author's surname must agree, and the journal must match.
A match is written with the metadata that Crossref returns (authors, title, journal, year, volume, issue,
pages, DOI), never with the remembered values. When author, year and journal agree but the title does not
(a missing subtitle, or a title remembered wrongly), the record is `title_differs` and stays `not_verified`
until a hand check in paper/references_manual.csv confirms it. Candidates without a confirmed record are
left out of refs.bib.

Crossref's record is the source of truth for the bibliographic fields. Whether a paper says what we
attribute to it is a separate check: the `content_checked` column records what was read (for example the
abstract on the publisher's page), and it is filled in by hand in references_manual.csv.

Run: python scripts/verify_references.py [--refresh]
Raw Crossref answers are cached in data/crossref/ (not committed) so reruns do not query again.
"""

from __future__ import annotations

import argparse
import difflib
import html
import json
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests

API = "https://api.crossref.org/works"
HEADERS = {"User-Agent": "keyperson-reference-check/0.1 (research script; python-requests)"}

# key, first author's surname, year, title, journal (as remembered; Crossref decides what is written)
CANDIDATES = [
    # Executive deaths, health, key human capital, founders, misconduct
    ("johnson1985", "Johnson", 1985, "An analysis of the stock price reaction to sudden executive deaths: Implications for the managerial labor market", "Journal of Accounting and Economics"),
    ("worrell1986", "Worrell", 1986, "Management turnover through deaths of key executives: Effects on investor wealth", "Academy of Management Journal"),
    ("salas2010", "Salas", 2010, "Entrenchment, governance, and the stock price reaction to sudden executive deaths", "Journal of Banking & Finance"),
    ("nguyen2010", "Nguyen", 2010, "The value of independent directors: Evidence from sudden deaths", "Journal of Financial Economics"),
    ("nguyen2014", "Nguyen", 2014, "What death can tell: Are executives paid for their contributions to firm value?", "Management Science"),
    ("quigley2017", "Quigley", 2017, "Shareholder perceptions of the changing impact of CEOs: Market reactions to unexpected CEO deaths, 1950-2009", "Strategic Management Journal"),
    ("bennedsen2020", "Bennedsen", 2020, "Do CEOs matter? Evidence from hospitalization events", "The Journal of Finance"),
    ("israelsen2017", "Israelsen", 2017, "Key human capital", "Journal of Financial and Quantitative Analysis"),
    ("fahlenbrach2009", "Fahlenbrach", 2009, "Founder-CEOs, investment, and firm value", "Journal of Financial and Quantitative Analysis"),
    ("cline2018", "Cline", 2018, "The consequences of managerial indiscretions: Sex, lies, and firm value", "Journal of Financial Economics"),
    ("borelli2021", "Borelli-Kjaer", 2021, "#MeToo: Sexual harassment and company value", "Journal of Corporate Finance"),
    ("neyland2020", "Neyland", 2020, "Love or money: The effect of CEO divorce on firm risk and compensation", "Journal of Corporate Finance"),
    # CEO turnover announcements, investigations, reputation
    ("warner1988", "Warner", 1988, "Stock prices and top management changes", "Journal of Financial Economics"),
    ("huson2004", "Huson", 2004, "Managerial succession and firm performance", "Journal of Financial Economics"),
    ("blackburne2021", "Blackburne", 2021, "Undisclosed SEC investigations", "Management Science"),
    ("johnson2014", "Johnson", 2014, "Corporate fraud and the value of reputations in the product market", "Journal of Corporate Finance"),
    ("karpoff2008", "Karpoff", 2008, "The cost to firms of cooking the books", "Journal of Financial and Quantitative Analysis"),
    # CEO power and boards
    ("finkelstein1992", "Finkelstein", 1992, "Power in top management teams: Dimensions, measurement, and validation", "Academy of Management Journal"),
    ("adams2005", "Adams", 2005, "Powerful CEOs and their impact on corporate performance", "Review of Financial Studies"),
    ("bebchuk2011", "Bebchuk", 2011, "The CEO pay slice", "Journal of Financial Economics"),
    ("morse2011", "Morse", 2011, "Are incentive contracts rigged by powerful CEOs?", "The Journal of Finance"),
    ("coles2014", "Coles", 2014, "Co-opted boards", "Review of Financial Studies"),
    # Executive networks
    ("fracassi2012", "Fracassi", 2012, "External networking and internal firm governance", "The Journal of Finance"),
    ("hwang2009", "Hwang", 2009, "It pays to have friends", "Journal of Financial Economics"),
    ("engelberg2013", "Engelberg", 2013, "The price of a CEO's Rolodex", "Review of Financial Studies"),
    ("elkhatib2015", "El-Khatib", 2015, "CEO network centrality and merger performance", "Journal of Financial Economics"),
    ("larcker2013", "Larcker", 2013, "Boardroom centrality and firm performance", "Journal of Accounting and Economics"),
    # Crisis response
    ("bundy2015", "Bundy", 2015, "A burden of responsibility: The role of social approval at the onset of a crisis", "Academy of Management Review"),
    ("coombs2007", "Coombs", 2007, "Protecting organization reputations during a crisis: The development and application of situational crisis communication theory", "Corporate Reputation Review"),
    ("lee2004", "Lee", 2004, "Mea culpa: Predicting stock prices from organizational attributions", "Personality and Social Psychology Bulletin"),
    # Pre-announcement run-ups
    ("meulbroek1992", "Meulbroek", 1992, "An empirical analysis of illegal insider trading", "The Journal of Finance"),
    ("keown1981", "Keown", 1981, "Merger announcements and insider trading activity: An empirical investigation", "The Journal of Finance"),
    # Media
    # Price discovery by session
    ("barclay1993", "Barclay", 1993, "Stealth trading and volatility: Which trades move prices?", "Journal of Financial Economics"),
    ("barclay2003", "Barclay", 2003, "Price discovery and trading after hours", "Review of Financial Studies"),
    ("tetlock2007", "Tetlock", 2007, "Giving content to investor sentiment: The role of media in the stock market", "The Journal of Finance"),
    ("fang2009", "Fang", 2009, "Media coverage and the cross-section of stock returns", "The Journal of Finance"),
    # Data
    ("gentry2021", "Gentry", 2021, "A database of CEO turnover and dismissal in S&P 1500 firms, 2000-2018", "Strategic Management Journal"),
    # Event-study method and inference
    ("mackinlay1997", "MacKinlay", 1997, "Event studies in economics and finance", "Journal of Economic Literature"),
    ("brown1985", "Brown", 1985, "Using daily stock returns: The case of event studies", "Journal of Financial Economics"),
    ("gelbach2013", "Gelbach", 2013, "Valid inference in single-firm, single-event studies", "American Law and Economics Review"),
    ("boehmer1991", "Boehmer", 1991, "Event-study methodology under conditions of event-induced variance", "Journal of Financial Economics"),
    ("kolari2010", "Kolari", 2010, "Event study testing with cross-sectional correlation of abnormal returns", "Review of Financial Studies"),
    ("patell1976", "Patell", 1976, "Corporate forecasts of earnings per share and stock price behavior: Empirical test", "Journal of Accounting Research"),
    # Jumps
    ("merton1976", "Merton", 1976, "Option pricing when underlying stock returns are discontinuous", "Journal of Financial Economics"),
]


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    text = re.sub(r"<[^>]+>", " ", text)  # Crossref titles sometimes carry markup
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalise(a), normalise(b)).ratio()


def query(title: str, author: str, cache: Path, refresh: bool) -> dict:
    path = cache / (re.sub(r"[^a-z0-9]+", "_", (author + "_" + title).lower())[:120] + ".json")
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    params = {"query.bibliographic": f"{title} {author}", "rows": 5}
    for wait in (5, 20, 60, 0):  # Crossref answers 429 when queried too fast
        response = requests.get(API, params=params, headers=HEADERS, timeout=60)
        if response.status_code != 429 or not wait:
            break
        time.sleep(wait)
    response.raise_for_status()
    data = response.json()
    path.write_text(json.dumps(data))
    time.sleep(0.5)
    return data


def year_of(item: dict) -> int | None:
    for field in ("published-print", "published", "issued", "published-online"):
        parts = item.get(field, {}).get("date-parts")
        if parts and parts[0] and parts[0][0]:
            return int(parts[0][0])
    return None


def authors_of(item: dict) -> list[str]:
    out = []
    for a in item.get("author", []):
        family, given = a.get("family", ""), a.get("given", "")
        out.append(f"{family}, {given}".strip(", ") if family else a.get("name", ""))
    return out


def best_match(candidate: tuple, items: list[dict]) -> tuple[dict | None, dict]:
    key, surname, year, title, journal = candidate
    best, best_checks = None, {}
    for item in items:
        if item.get("type") not in ("journal-article", None):
            continue
        ctitle = (item.get("title") or [""])[0]
        cjournal = (item.get("container-title") or [""])[0]
        cyear = year_of(item)
        first = authors_of(item)[0] if item.get("author") else ""
        checks = {
            "title_similarity": round(similarity(title, ctitle), 3),
            "journal_similarity": round(similarity(journal.replace("The ", ""), cjournal.replace("The ", "")), 3),
            "year_crossref": cyear,
            "first_author_crossref": first,
            "year_ok": cyear is not None and abs(cyear - year) <= 1,
            "author_ok": normalise(surname.split("-")[0]) in normalise(first),
        }
        score = checks["title_similarity"] + 0.5 * checks["journal_similarity"] + 0.3 * checks["year_ok"] + 0.3 * checks["author_ok"]
        if best is None or score > best_checks["score"]:
            best, best_checks = item, checks | {"score": round(score, 3)}
    return best, best_checks


def clean(text) -> str:
    """Crossref strings sometimes carry markup, HTML entities, footnote marks or stray spaces."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = html.unescape(re.sub(r"<[^>]+>", "", str(text))).replace("\u2606", "")
    return re.sub(r"\s+:", ":", re.sub(r"\s+", " ", text)).strip()


def latex(text: str) -> str:
    return re.sub(r"(?<!\\)([&#%])", r"\\\1", text)


def bibtex(row: dict) -> str:
    authors = " and ".join(a.strip() for a in str(row["authors"]).split(";"))
    fields = {
        "author": authors,
        "title": row["title"],
        "journal": row["journal"],
        "year": str(row["year"]),
        "volume": row["volume"],
        "number": row["issue"],
        "pages": str(row["pages"]).replace("-", "--").replace("\u2013", "--"),
        "doi": row["doi"],
        "url": row.get("url", ""),
    }
    body = ",\n".join(f"  {k} = {{{latex(str(v)) if k not in ('doi', 'url') else v}}}"
                      for k, v in fields.items() if str(v) not in ("", "nan"))
    return f"@article{{{row['key']},\n{body}\n}}\n"


BIB_FIELDS = ["authors", "title", "journal", "year", "volume", "issue", "pages", "doi"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="query Crossref again instead of using the cache")
    parser.add_argument("--out", default="paper")
    parser.add_argument("--manual", default="paper/references_manual.csv",
                        help="hand checks: key, status, corrected fields, url, extra_source_checked, content_checked, notes")
    args = parser.parse_args()
    cache = Path("data/crossref")
    cache.mkdir(parents=True, exist_ok=True)
    manual = pd.read_csv(args.manual, dtype=str).set_index("key") if Path(args.manual).exists() else pd.DataFrame()

    rows, entries = [], []
    for candidate in CANDIDATES:
        key, surname, year, title, journal = candidate
        data = query(title, surname, cache, args.refresh)
        item, checks = best_match(candidate, data.get("message", {}).get("items", []))
        # same record: first author, year (+-1) and journal agree; a full match also needs the title to agree
        same = bool(item) and checks["year_ok"] and checks["author_ok"] and checks["journal_similarity"] >= 0.6
        auto_ok = same and checks["title_similarity"] >= 0.85
        row = {"key": key, "status": "verified" if auto_ok else "not_verified",
               "status_crossref_match": "match" if auto_ok else ("title_differs" if same else "no_match")}
        row |= dict.fromkeys(BIB_FIELDS, "")
        if same:
            row |= {
                "authors": "; ".join(authors_of(item)),
                "title": clean((item.get("title") or [""])[0]),
                "journal": clean((item.get("container-title") or [""])[0]),
                "year": year_of(item),
                "volume": item.get("volume", ""),
                "issue": item.get("issue", ""),
                "pages": item.get("page", ""),
                "doi": item.get("DOI", ""),
            }
        row |= {"url": "", "source_checked": "Crossref API (query.bibliographic, best of 5 results)",
                "extra_source_checked": "", "content_checked": "none: bibliographic metadata only", "notes": ""}
        row |= {"candidate_first_author": surname, "candidate_year": year, "candidate_title": title,
                "candidate_journal": journal}
        row |= {k: v for k, v in checks.items() if k != "score"}
        row["crossref_best_doi"] = item.get("DOI", "") if item else ""
        if key in manual.index:
            for column, value in manual.loc[key].items():
                if isinstance(value, str) and value.strip():
                    row[column] = value.strip()
        rows.append(row)
        if row["status"] == "verified":
            entries.append(bibtex(row))
    table = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    table.to_csv(out / "references_verification.csv", index=False)
    # News items and filings (no DOI) are kept by hand in paper/refs_news.bib and appended.
    news = out / "refs_news.bib"
    (out / "refs.bib").write_text("\n".join(entries) + ("\n" + news.read_text() if news.exists() else ""))
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(table[["key", "status", "status_crossref_match", "title_similarity", "year_crossref", "doi"]].to_string(index=False))
    print(f"\n{sum(table['status'] == 'verified')} verified, {sum(table['status'] != 'verified')} not verified")


if __name__ == "__main__":
    main()
