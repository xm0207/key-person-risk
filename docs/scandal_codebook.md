# Coding CEO personal-scandal candidates (events/ceo_scandal_coding.csv)

The candidates come from `scripts/build_scandal_candidates.py`:
- **Source stories:** Dow Jones stories whose headline names the firm's sitting CEO and contains a scandal word.
- **Episodes:** the first story after at least 365 days without one starts a new episode.

Each episode is coded from its headlines in the first 30 days. The headlines themselves are Dow Jones content, so they stay in `data/derived/` (gitignored).

**Second pass.** A stricter word list, matched on whole words only (so that "Semiconductor" no longer matches "conduct" and "Pleased" no longer matches "plea"), found episodes that start later than the first pass did. These are coded with an `S` prefix. Example: a CEO's resignation over a relationship that came five months after an unrelated law-firm notice.

| Code | Meaning |
|---|---|
| P1 | Serious personal-conduct allegation against the sitting CEO, meaning conduct outside the CEO's job at the firm: a relationship or sexual conduct, harassment, a crime or personal legal charge (including trading in another firm's shares), dishonesty about the CEO's own record, personal dealings with the firm or misuse of its money, or a board investigation of such conduct |
| P2 | A gaffe or behavioural controversy: offensive remarks, tweets, a public apology for the CEO's own words |
| W | The CEO's conduct in running the firm: accounting fraud, option backdating, bribery, conflicts in a sale of the firm, and trading in or statements about the firm's own securities |
| F | A firm-level legal or regulatory problem in which the CEO is named only as head of the firm, or a controversy over pay or performance |
| H | Health: a medical leave or a death |
| U | Uncertain: the headlines do not show what the allegation is; needs checking against the full story |
| N | Not a scandal: a wrong match, a routine announcement, a word match inside another word |
| DUP | The same matter as another candidate |

**Revision of 18 November 2026.** A check of the case dates against SEC filings and news reports (via the author's library access) led to these changes:
- **Recoded P1 to W:** C0025, C0048, C0288 and S0081. Each concerns trading in, or statements about, the firm's own securities, or the CEO's role in the firm's research. This follows the line drawn by Cline, Walkling and Yore (2018), who leave securities violations out of "indiscretions".
- **Recoded P1 to U:** C0317. The board put the CEO on leave over "corporate policies" and never said what the conduct was.
- **Bug fixed in the search:** a RavenPack entity linked to two gvkeys (for example Activision Inc. and Activision Blizzard) kept only the first. The fix adds C0339 to C0341 (coded N, W, F) and S0115 (Activision Blizzard, November 2021, coded P1). Candidate IDs are now kept stable when the search is re-run (`scandals.stable_ids`).
- **Second-pass word list:** restored to the list that produced the coded file; it does not include "leave".
- **Case added by hand:** M0001 (Yahoo, 2012), listed in `events/personal_scandal_additions.csv`. The RavenPack link uses the current Compustat CUSIP, so it misses firms whose CUSIP changed.
- **First-news dates moved earlier:** Chesapeake to 18 April 2012 and Whole Foods to 12 July 2007 (`events/personal_scandal_overrides.csv`).

**Validation.** Before the codes are used, the author should recode a random 10% of candidates without seeing these codes, and report the agreement rate (Cohen's kappa).

## First-pass counts (17 Nov 2026)
- **Candidates:** 338, plus 6 from the second pass.
- **Codes:**

| P1 | P2 | W | F | H | U | N | DUP |
|---|---|---|---|---|---|---|---|
| 32 | 10 | 12 | 51 | 34 | 10 | 192 | 3 |

After the 18 November revision, the 345 coded candidates (341 first pass, plus 4 second pass) split as: P1 28, P2 10, W 17, F 52, H 34, U 11, N 193, DUP 3.

- **Outcomes of the 32 P1 episodes:**
  - **18 CEOs left within a year,** according to Gentry et al. Thirteen of the 18 exits were announced within three days of the first story: the scandal became public together with the exit.
  - **About 8 CEOs stayed.**
  - **The rest** are after Gentry's coverage ends; their outcome comes from the headlines.
