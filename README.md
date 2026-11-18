# CEO scandals and stock prices: Nvidia in 2026 and 79 earlier cases

All numbers below are Nvidia's own share-price moves: its return minus what the S&P 500 and the six other largest US tech stocks explain. Prices run to the close of 17 November 2026.

## What happened to Nvidia's share price

**1. The day the CEO resigned.**
- On closing prices, Nvidia fell 3.5% on 12 November. Counting from the last trade the night before, it rose 3.8%.
- New allegations came out after the close on 11 November, and the stock fell 7.0% in after-hours trading.
- The resignation was filed at 8:52 am on 12 November. The stock fell 5.4% before the open and rose 8.8% in regular trading.

**2. Most of the moves happened while the main market was closed.** During the three weeks, 69% of Nvidia's own price moves happened before 9:30 am or after 4 pm. In the two months before, the figure was 26–36%.
- The first story came out at 5 am.
- The board's statement came on a Sunday.
- The new allegations came after the close.
- The resignation came before the open.

**3. The board's statement moved the stock more than the newspaper story.**
- After the first newspaper report, Nvidia's own move was about −1%.
- After the board announced an investigation, it was −13.3% over two trading days, about $600 billion.

**4. The rise at the departure has happened before.** Of 79 personal-conduct scandals involving US CEOs since 2000, 8 became public before the CEO left. In 7 of those 8, the stock rose when the CEO went.

The working paper, with the details of every case, is [paper/personal_scandals.pdf](paper/personal_scandals.pdf).

## What is here

- `paper/`: the paper (PDF and LaTeX), its tables and figure.
- `scripts/` and `src/keyperson/`: the full pipeline, from the data pulls to the tables, run with `make all`.
- `events/`: the hand-coded inputs (case codes, date corrections with their sources, the Nvidia timeline with sources).
- `results/`: summary statistics behind every table and every number in the text.

The WRDS, Dow Jones and Yahoo Finance data are licensed or for personal use and are not included. The pipeline rebuilds them for anyone with access.

## Reproducing the paper

The Makefile expects the virtual environment at `~/.venvs/key-person-risk`.

```bash
uv venv --python 3.12 ~/.venvs/key-person-risk
cd "/path/to/Projects/key-person-risk"
uv pip install --python ~/.venvs/key-person-risk/bin/python -e ".[dev]"   # also installs mega7 from GitHub

make wrds      # once: WRDS extracts (asks for your WRDS login; about an hour)
make prices    # once: Yahoo Finance prices for the Nvidia case
make all       # sample -> analysis -> tables -> text numbers -> PDF -> tests
```

`make all` runs every step after the data pulls and reproduces every file in `results/` and `paper/tables/`.

### What you need

| Source | What | Where it goes |
|---|---|---|
| WRDS (subscription) | CRSP daily stock file and names, Compustat (annual and quarterly), ExecuComp, the CRSP/Compustat link, Fama–French factors, RavenPack (Dow Jones edition) | `data/wrds/`. The web-query extracts are listed in `data/wrds/MANIFEST.md`; the rest come from `make wrds` |
| Zenodo | Gentry, Harrison, Quigley and Boivie (2021) CEO dismissal database, version of 10 October 2024 ([10.5281/zenodo.17725939](https://doi.org/10.5281/zenodo.17725939)), CC BY 4.0 | `data/external/gentry/` |
| Yahoo Finance | daily closes for Nvidia, the S&P 500 (SPY), six other mega-caps and sector funds, to 17 November 2026; daily and hourly bars with extended hours for Nvidia, SPY and the peers | `data/keyperson.duckdb`, `data/yahoo_sessions/` |
| Crossref | bibliographic records of the references | `data/crossref/` (cache) |

Nothing in `data/` is committed.

## The pipeline

| Step | Script | Reads | Writes | Used in the paper |
|---|---|---|---|---|
| Data | `wrds_pull_events.py` | RavenPack executive events and company mappings | `data/wrds/ravenpack/` | sample |
| | `wrds_pull_stage3.py` | CRSP windows for the event sample, Fama–French factors, RavenPack company news | `data/wrds/stage3/` | departure study, press coverage |
| | `wrds_pull_stage4.py` | Compustat earnings dates, RavenPack CEO-scandal headlines, CRSP 1999–2024 for ExecuComp firms | `data/wrds/stage4/` | candidates, scandal returns |
| | `download_data.py` | Yahoo Finance | `data/keyperson.duckdb` | Nvidia |
| Sample | `build_event_sample.py` | Gentry database, CCM link, Compustat, RavenPack mappings | `data/derived/event_sample.parquet` | all CEO departures |
| | `build_event_features.py` | event sample, ExecuComp, Compustat, RavenPack | `data/derived/event_features.parquet` | departure study |
| | `build_scandal_candidates.py` | stage-4 headlines, ExecuComp | `data/derived/scandal_candidates*.csv` | hand coding |
| Analysis | `run_departure_study.py` | event sample, features, stage-3 CRSP | `data/derived/departure_windows.parquet`, `results/departure_*.csv` | personal against corporate departures (Section 5) |
| | `run_personal_scandals.py` | sample, hand codes, date corrections, stage-4 CRSP | `data/derived/personal_scandals.parquet`, `results/personal_scandal_*.csv` | Sections 5–7, Appendix B |
| | `nvidia_timeline.py` | Yahoo prices | `results/nvidia_timeline.csv` | Section 2 |
| | `nvidia_sessions.py` | Yahoo daily and hourly bars with extended hours | `results/nvidia_sessions.csv`, `results/nvidia_session_days.csv` | Section 2.1, Prediction 8 |
| Paper | `paper_assets.py` | the outputs above | `paper/tables/*.tex`, `paper/fig_paths.pdf` | Tables 1–6, Figure 1 |
| | `paper_numbers.py` | the outputs above, and the .tex | `results/paper_text_numbers.csv` | every number quoted in the text but not in a table |
| | `verify_references.py` | Crossref | `paper/refs.bib`, `paper/references_verification.csv` | references |

`paper_numbers.py` also checks that each number appears in the paper exactly as quoted and prints any that do not. A change in the data therefore cannot leave a stale number in the text unnoticed.

## Hand-coded inputs (`events/`)

These are the judgements the paper rests on, kept separate from the code so they can be checked:

- `gentry_misconduct_coding.csv`: each Gentry code-4 departure classified as personal conduct, corporate wrongdoing, or other. The rule: personal conduct is conduct outside the chief executive's job at the firm, and trading in the firm's own shares counts as corporate wrongdoing.
- `ceo_scandal_coding.csv`: each headline candidate coded P1 (serious personal-conduct allegation), P2, W, F, H, U, N or DUP. It holds identifiers and codes only; the headlines are Dow Jones content and stay in `data/derived/`. The codebook is `docs/scandal_codebook.md`.
- `personal_scandal_overrides.csv`: first-news and exit dates corrected by hand, each with its evidence (an SEC filing, a release or a dated story).
- `personal_scandal_additions.csv`: cases the search cannot find (Yahoo 2012, whose RavenPack link breaks on a CUSIP change).
- `founder_manual.csv`: founder status checked by hand for the two-stage and survival cases.
- `nvidia_timeline_sources.csv`: date, time and source of each Nvidia event, October–November 2026.
- `megacap_events.csv`: CEO-level events at seven mega-caps, from the project's first design (see `archive/`).

## References

`scripts/verify_references.py` queries Crossref for each reference and writes Crossref's metadata, not remembered values; all 42 were confirmed. Hand checks against publisher records are in `paper/references_manual.csv`. Its `content_checked` column records what was actually read:

- **Full texts** (through the author's library access, 17 November 2026): Cline, Walkling and Yore (2018), Borelli-Kjaer et al. (2021), Adams, Almeida and Ferreira (2005), Fahlenbrach (2009), Warner, Watts and Wruck (1988), Huson, Malatesta and Parrino (2004), Neyland (2020), and Johnson, Xie and Yi (2014).
- **Abstracts only:** Finkelstein (1992), Fang and Peress (2009), Blackburne et al. (2021), Gelbach, Helland and Klick (2013), MacKinlay (1997).
- **Data documentation:** Gentry et al. (2021).

The paper describes each paper only at the level that was read. News items and filings, which have no DOI, are in `paper/refs_news.bib`, which the script appends to `refs.bib`.

Crossref left some fields wrong or missing; the corrections, filled from RePEc and checked by hand, were:
- Fahlenbrach (2009) is titled "Founder-CEOs, Investment Decisions, and Stock Market Performance".
- The third author of the #MeToo paper is Ulf Nielsson.
- MacKinlay (1997) has no DOI.
- Crossref omits some subtitles, lists only the first author of Boehmer et al. (1991), and gives only the first page of Patell (1976).

## The event-study library (`src/keyperson/`)

The analysis uses a general event-study engine that works with any ticker and list of event dates (`engine.py`). It is built on the market model of [mega7-decomposition](https://github.com/xm0207/mega7-decomposition), with an optional sector factor orthogonalised to the market. Other modules:

- the Gelbach–Helland–Klick single-firm SQ test (`sq_test.py`);
- cross-sectional tests (`cross_section.py`);
- priced-in and post-event measures (`measures.py`);
- day-0 rules (`dates.py`);
- CRSP handling (`crsp.py`);
- sample construction and linking (`sample.py`);
- CEO and firm power (`power.py`);
- the headline search (`scandals.py`);
- WRDS helpers (`wrdsio.py`).

`docs/engine.md` describes the estimation rules and tests in detail. `tests/` is an offline pytest suite (`make test`).

## Data use

Yahoo Finance data are for personal use. WRDS data (CRSP, Compustat, ExecuComp, RavenPack) are licensed and are not redistributed. Dow Jones headlines stay in `data/derived/`, which is not committed.

`results/` and `archive/results/` hold event-level summary statistics only. A test applies mega7's rule that no committed table may look like a daily or weekly series.

## Layout

```
paper/           personal_scandals.tex and .pdf, tables/, fig_paths.pdf, refs.bib, refs_news.bib, reference checks
scripts/         the pipeline above, in the order of the Makefile
src/keyperson/   the event-study library
events/          hand-coded inputs
results/         summary statistics behind the paper, and paper_text_numbers.csv
docs/            scandal_codebook.md and engine.md; docs/archive/ holds superseded working notes
archive/         earlier phases of the project and checks not used in the paper (see archive/README.md)
tests/           offline test suite
Makefile         make all, make wrds, make prices
```

## Licence

MIT for the code. Event descriptions cite their sources.
