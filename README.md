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

- **`METHODS.md`** — methods, splits, results, reproducibility, and the traps
  hit along the way. Read it before running anything or quoting any number.
- Writeup: *When can you trust an ML streamflow forecast?* (personal site).

## Layout

| Path | What |
|---|---|
| `configs/` | Training/eval configs for all runs (original split, flood split, CMAL, leave-one-basin-out) |
| `extend_targets.py` | Streamflow record extension 2015–2024 from USGS NWIS, with closure test |
| `make_baselines.py` | Honest-baseline suite: mean, climatology, persistence (by lead) |
| `make_lstm_by_lead.py` | Per-lead LSTM skill on the flood window |
| `make_benchmark.py` | NWM v2.1 benchmark, original 2012–2014 window |
| `make_benchmark_flood.py` | NWM v2.1/v3.0 benchmark on the flood years |
| `make_probabilistic_scores.py` | CRPS, coverage, reliability, spread/skill |
| `make_model_comparison.py` | The headline model-comparison table: one protocol, fixed cohort |
| `make_cmal_point_metrics.py` | CMAL point metrics from stored samples, with bootstrap spread |
| `make_pit_diagnostic.py` | PIT/rank histogram + flow-conditional coverage |
| `make_figures_flood.py` | Figure suite for the writeup |
| `make_1997.py` | Held-out January-1997 flood analysis |
| `make_figures.py` | Hydrograph/skill figures |
| `upstream/` | Pinned-commit setup for the vendored framework + local patch |

## Setup

1. `upstream/setup_upstream.sh` clones the framework at the pinned commit and
   applies the local patch (anonymous GCS reads).
2. Build the env from `upstream/conda-macos.yml` (macOS; see METHODS.md §1 for
   the micromamba route and the CUDA-only pitfall in upstream's own env file).
3. `pip install -e central_valley_floodforecasting`.
4. Data: streamflow targets from Caravan-nc (Zenodo) extended via
   `extend_targets.py`; forcing streams from `gs://caravan-multimet/v1.1`.
   Nothing large is stored in this repo.

Configs carry absolute paths from the machine this was built on; point
`run_dir`, basin-file paths, and the data dirs at your own locations.
