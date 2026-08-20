# California streamflow LSTM — project context

Daily streamflow forecasting on 28 California basins (20 rain / 8 snow by frac_snow;
North Coast + Central Coast + Sierra/Tahoe — NOT a Central Valley or Sierra-snowmelt
cohort) with Google Research's open flood-forecasting framework. Five-findings writeup,
headlined by the gauge-persistence crossover. Public repo:
github.com/holdenlesliebole/central-valley-flood-lstm.

**Status: FROZEN as of 2026-08-19.** An adversarial review (`ADVERSARIAL_REVIEW.md`,
local-only, untracked) found a multi-lead valid-date scoring bug; corrections are in
`docs/adversarial_review_tier1_memo.md` (committed) and the regenerated outputs.
No further research investment: no retraining, no new experiments, no capacity sweeps.
The review's FIRO/inflow-ensemble direction is a possible future project, not
remediation. Reopen only by Holden's explicit decision.

## Read this first

- **`docs/METHODS.md` §4 is the sole numeric source of truth.** Its §5 traps are load-
  bearing — read §5 before running anything or quoting any number.
- The public-facing writeup lives outside this repo:
  `~/Documents/Job_Search/portfolio/central_valley_writeup.md` (markdown master) and the
  site note `PersonalWebsite/.../_notes/central-valley-flood-lstm.md`. `paper/main.tex`
  here is the LaTeX rendition for reading/proofing. **Any edit to results or prose must
  land in all three or be flagged as pending sync.**

## Layout

| Path | What |
|---|---|
| `docs/METHODS.md` | Methods, results, traps — numeric source of truth |
| `paper/main.tex` | LaTeX manuscript (build: pdflatex/bibtex chain from `paper/`) |
| `scripts/` | All analysis + figure scripts; run from repo root: `python scripts/make_<x>.py` |
| `configs/` | Training/eval configs (original split, flood split, CMAL, LOBO, orig-h128) |
| `outputs/figures/` | Canonical figures + metric CSVs (committed) |
| `runs/` | Trained runs (untracked; multi-GB) |
| `upstream/` | Pinned upstream reconstruction (`setup_upstream.sh` + patch) |
| `central_valley_floodforecasting/` | Vendored upstream clone (untracked; rebuild via upstream/) |

Data stores live outside the repo: `~/data/caravan-nc-extended` (targets),
`~/data/nwis_cache`, `~/data/nwm*_focus*.nc` (NWM caches); forcing streams from GCS.

## How to run

```bash
conda activate googlehydrology
cd ~/Documents/Side_projects/Hydrology
export TORCHDYNAMO_DISABLE=1     # mandatory on this machine (METHODS §1.2)
env -u GOOGLE_APPLICATION_CREDENTIALS CLOUDSDK_CONFIG=/tmp/empty_gcloud \
  run train --config-file configs/<name>.yml
```

Hard rules (details in METHODS §5): use `run infer`, never `run evaluate`, for metrics
(evaluate silently restricts the cohort AND deletes test_results.zarr); never trust obs
from a CMAL zarr — score stored samples against source netCDFs; one process on MPS at a
time; state the 22-basin cohort with every median.
