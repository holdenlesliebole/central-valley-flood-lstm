# Central Valley flood-forecasting LSTM

California Sierra streamflow forecasting with Google Research's open
[flood-forecasting framework](https://github.com/google-research/flood-forecasting)
(the LSTM behind Flood Hub), trained and evaluated on a laptop across 28
California basins and benchmarked against the NOAA National Water Model
retrospective on held-out flood water years (WY2017, WY2023).

The question this project actually answers is not "does the LSTM win" but
**when can you trust an ML streamflow forecast** — test-period choice, gauge
persistence as the binding baseline, peak-magnitude skill, ungauged transfer
across hydrologic regimes, and probabilistic calibration.

- **`docs/METHODS.md`** — methods, splits, results, reproducibility, and the traps
  hit along the way. Read it before running anything or quoting any number.
- Writeup: *When can you trust an ML streamflow forecast?* (personal site).

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
