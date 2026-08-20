# California streamflow LSTM: forecast value against the gauge

Daily streamflow forecasting across 28 minimally regulated California basins
(20 rain-classified, 8 snow) with Google Research's open
[flood-forecasting framework](https://github.com/google-research/flood-forecasting)
(the same model family as Flood Hub), trained and evaluated on a laptop and
scored on held-out flood water years (WY2017, WY2023) against the baselines an
operator has in hand: lead-matched gauge persistence, damped persistence, and
NOAA National Water Model retrospectives.

The question this project answers is when a deep-learning streamflow forecast
beats the gauge reading the operator already has, and where its value stops:
storm-stratified skill, peak-magnitude error, a held-out-basin failure,
and probabilistic calibration before and after a cross-fitted affine
correction.

**2026-08-19 correction.** An adversarial review found that two multi-lead
analyses paired forecasts with observations from the issue date rather than
the valid date. The recalibration and lead-3 storm results were regenerated,
a per-lead observation closure test was added
(`scripts/test_valid_date_alignment.py`), and the split-boundary label overlap
was quantified (`outputs/figures/split_leakage.csv`). Details in
`docs/METHODS.md`.

- **`docs/METHODS.md`** — methods, splits, results, reproducibility, and the traps
  hit along the way. Read it before running anything or quoting any number.
- Writeup: *When does a deep-learning streamflow forecast beat the gauge?*
  (personal site).

## Layout

| Path | What |
|---|---|
| `docs/METHODS.md` | Methods, results, reproducibility, traps — the numeric source of truth |
| `paper/` | LaTeX manuscript of the writeup (`main.tex`) |
| `scripts/` | All analysis and figure scripts (record extension, baselines, benchmarks, scoring, figures) |
| `configs/` | Training/eval configs for all runs (original split, flood split, CMAL, leave-one-basin-out) |
| `outputs/figures/` | Canonical figures and metric CSVs, written by `scripts/` |
| `runs/` | Trained runs (local only, not tracked) |
| `upstream/` | Pinned-commit setup for the vendored framework + local patch |

## Setup

1. `upstream/setup_upstream.sh` clones the framework at the pinned commit and
   applies the local patch (anonymous GCS reads).
2. Build the env from `upstream/conda-macos.yml` (macOS; see METHODS.md §1 for
   the micromamba route and the CUDA-only pitfall in upstream's own env file).
3. `pip install -e central_valley_floodforecasting`.
   Run scripts from the repo root: `python scripts/make_<name>.py`.
4. Data: streamflow targets from Caravan-nc (Zenodo) extended via
   `extend_targets.py`; forcing streams from `gs://caravan-multimet/v1.1`.
   Nothing large is stored in this repo.

Configs carry absolute paths from the machine this was built on; point
`run_dir`, basin-file paths, and the data dirs at your own locations.
