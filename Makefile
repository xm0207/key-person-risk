# Reproduce the paper. Run from the project folder: `make all` rebuilds everything after the data pulls.
# The WRDS pulls (`make wrds`) ask for your WRDS login and take about an hour; run them once.
# Yahoo prices for the Nvidia case: `make prices`. Nothing here writes outside data/, results/ and paper/.

PY ?= $(HOME)/.venvs/key-person-risk/bin/python
LATEX = pdflatex -interaction=nonstopmode -halt-on-error personal_scandals.tex > /dev/null

.PHONY: all wrds prices sample analysis tables numbers refs pdf test

all: sample analysis tables numbers pdf test

wrds:      ## WRDS extracts -> data/wrds/ (the event sample is needed to choose the CRSP windows of stage 3)
	$(PY) scripts/wrds_pull_events.py
	$(PY) scripts/build_event_sample.py
	$(PY) scripts/wrds_pull_stage3.py
	$(PY) scripts/wrds_pull_stage4.py

prices:    ## Yahoo Finance daily closes -> data/keyperson.duckdb, and daily and hourly bars -> data/yahoo_sessions/
	$(PY) scripts/download_data.py
	$(PY) scripts/nvidia_sessions.py --refresh

sample:    ## departures, their features, and the headline candidates
	$(PY) scripts/build_event_sample.py
	$(PY) scripts/build_event_features.py
	$(PY) scripts/build_scandal_candidates.py

analysis:  ## event studies -> results/
	$(PY) scripts/run_departure_study.py
	$(PY) scripts/run_personal_scandals.py
	$(PY) scripts/nvidia_timeline.py
	$(PY) scripts/nvidia_sessions.py

tables:    ## paper/tables/*.tex and paper/fig_paths.pdf
	$(PY) scripts/paper_assets.py

numbers:   ## numbers quoted in the text -> results/paper_text_numbers.csv, checked against the .tex
	$(PY) scripts/paper_numbers.py

refs:      ## Crossref check of the references -> paper/refs.bib (cached answers in data/crossref/)
	$(PY) scripts/verify_references.py

pdf:
	cd paper && $(LATEX) && bibtex personal_scandals > /dev/null && $(LATEX) && $(LATEX)
	@! grep -n "undefined\|Overfull" paper/personal_scandals.log

test:
	$(PY) -m pytest -q
