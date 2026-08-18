# Central Valley flood-forecasting LSTM — methods, results, and traps

**Last updated 2026-08-17.** This is the reproducibility and methodology record for the
code in this repo; the public-facing summary is the site writeup, which must not drift
from the numbers here. Paths in this file are machine-specific.

Upstream is Google Research's open flood-forecasting framework (package
`googlehydrology`, forked from NeuralHydrology, vendored in
`central_valley_floodforecasting/`). This repo adds California-specific configs, a
record extension, an honest-baseline suite, and the evaluation protocol below.

---

## 1. Environment and how to run

```bash
source ~/opt/anaconda3/etc/profile.d/conda.sh
conda activate googlehydrology
cd ~/Documents/Side_projects/Hydrology
export TORCHDYNAMO_DISABLE=1                 # REQUIRED - see 1.2
run train    --config-file configs/<name>.yml
run evaluate --run-dir runs/<run> --epoch N --period test
run infer    --run-dir runs/<run> --epoch N --period test   # writes test_results.zarr
```

### 1.1 Four breakages from the 2026-07 repo move (all fixed 2026-08-11)

The project moved from `~/Documents/Hydrology` to `~/Documents/Side_projects/Hydrology`.
That broke four things at once, none of which announced itself clearly:

1. **Dead paths in `configs/*.yml`** (`run_dir`, `train/validation/test_basin_file`).
2. **Dead paths inside each run directory's own saved `config.yml`** — `run evaluate`
   reads the *run dir's* config, not the one in `configs/`. Easy to miss.
3. **Broken editable install.** `pip install -e .` still pointed at the old location, so
   `import googlehydrology` failed. Fix: reinstall from the new path.
4. **Expired Google application-default credentials** (`invalid_grant`).

### 1.2 `TORCHDYNAMO_DISABLE=1` is mandatory

Without it the model is wrapped in `torch.compile`'s `OptimizedModule`, whose state-dict
keys carry an `_orig_mod.` prefix. Existing checkpoints were saved uncompiled, so loading
fails with a wall of missing/unexpected keys. (Consistent with the standing note that
torch.compile/inductor cannot target MPS on this machine.)

### 1.3 GCS is read anonymously (local patch to vendored code)

`googlehydrology/datasetzoo/multimet.py::_open_zarr` was patched to pass
`storage_options={'token': 'anon'}` for `gs://` paths, overridable via
`GOOGLEHYDROLOGY_GCS_TOKEN`. Rationale: the Caravan-MultiMet bucket is public, relying on
application-default credentials made runs fail once the cached refresh token expired, and
requiring a Google account would make this work non-reproducible for anyone else.

---

## 2. Data

### 2.1 Sources

| What | Where | Coverage |
|---|---|---|
| Streamflow targets + static attributes | `~/data/caravan-nc` (Caravan-nc, local) | date axis to 2023-12-30, **but streamflow is all-NaN after 2014** |
| Extended targets | `~/data/caravan-nc-extended` (built here) | streamflow filled 2015-01-01 → 2024-10-31 |
| Meteorological forcing | `gs://caravan-multimet/v1.1` (streamed) | **ERA5-Land + IMERG end 2024-10-31**; HRES 2024-09-30; CPC 2024-07-31 |
| NWM v2.1 retrospective | AWS zarr, cached `~/data/nwm_focus_2012_2014.nc` | process-model baseline |
| USGS NWIS | web service, cached `~/data/nwis_cache` | present-day |

**The 2024 wall is the forcing, not the targets.** Caravan-MultiMet is a frozen versioned
snapshot, not a live service. USGS serves these gauges to the present. Extending past
Oct 2024 therefore requires building a forcing pipeline (ERA5-Land via Copernicus CDS, or
NOAA AORC) *plus* catchment zonal statistics — the basin-averaging is the expensive half.

### 2.2 Record extension — `extend_targets.py`

Fills streamflow 2015→2024-10-31 from USGS NWIS. **23 of 28 basins extended, median
+3,592 days. WY2017 and WY2023 both get a median 365 valid days.**

Unit conversion: `mm/day = cfs * 2.446575 / area_km2`.

**The closure test is not optional and caught two real problems.** Before writing, USGS
data is pulled over an overlap window (2012–2014) and compared against the existing
Caravan record.

1. **Correlations of 1.0000 against median relative errors of 0.16%–18.8%.** Perfect
   correlation with a systematic offset is a **scale factor** — Caravan's catchment area
   disagrees with the area implied by the USGS record. Fixed by fitting a per-basin factor
   from the overlap. *Which area is "truer" is not the question*: the model was trained on
   Caravan's convention, so the extension must be continuous with that.
2. **Guard against over-fixing.** A real scale factor gives a near-constant ratio, so
   basins with ratio IQR > 0.05 are **excluded, not rescaled**. Five fail this way
   (11124500, 11141280, 11151300, 11176400, 11284400; IQR 0.05–0.24) — genuinely
   disagreeing series, likely regulation or diversion.

**Side result:** the fitted factors bound the Caravan-area error directly. Bear Ck implied
area 135.9 km² vs Caravan 142.3 = **4.5%** — far too small to explain NWM's −0.47 NSE
there, so that result is not a conversion artifact.

---

## 3. Experimental design

### 3.1 Splits

**Original (2026-06):** train 1985–2008 · val 2009–2011 · test 2012–2014.
The test window is **CA drought onset**, which flatters skill and undertests peaks.

**Flood split (current, 2026-08-12)** — every day used, no gaps:

| Period | Use |
|---|---|
| 1985-01-01 → 2008-12-31 | train |
| 2009-01-01 → 2011-12-31 | validation (unchanged, so checkpoint selection stays comparable) |
| 2012-01-01 → 2016-09-30 | train |
| **2016-10-01 → 2017-09-30** | **TEST — WY2017** (Oroville-spillway AR season) |
| 2017-10-01 → 2022-09-30 | train |
| **2022-10-01 → 2023-09-30** | **TEST — WY2023** (extreme AR sequence) |
| 2023-10-01 → 2024-10-31 | train |

Training grows 24 → ~34.8 years. The framework accepts **lists** for
`train_start_date`/`train_end_date`/`test_*`, which is what makes disjoint periods possible.
**Consequence: 2012–2014 is now training data, so flood-split numbers are NOT comparable
to the original 0.808.** Both heads were retrained on this split for that reason.

### 3.2 Checkpoint selection — part of the method, not a detail

Use `validate_every: 1` and **evaluate the best-validation epoch**, never the last. The
original workflow evaluated the last epoch and got away with it by luck: the deterministic
model was still improving at epoch 15. CMAL at h=16 peaks at **epoch 2** and degrades
monotonically afterward while its training loss keeps improving — last-epoch evaluation
would have reported NSE 0.463 instead of its true best.

### 3.3 Metrics

`NSE, KGE, FHV, Peak-Timing, Missed-Peaks, Peak-MAPE`. The peak metrics ship with the
framework and were simply not requested by the original config. FHV (flow-duration high
segment) and Peak-MAPE are the operationally meaningful ones — average NSE hides peaks.

---

## 4. Results

### 4.1 Model comparison — REVISED 2026-08-14 under one protocol

**All numbers in this table come from `make_model_comparison.py`:** sims from each
run's `run infer` zarr, CMAL summarized by the per-day sample median (7500 samples),
observations from the source netCDFs, time_step 0, framework metric functions, and a
**fixed 22-basin cohort** (of 28 trained, 6 have no flood-window observations — see
trap §5.8: the earlier `run evaluate` numbers silently used a 16-basin cohort).
Flood-split test = WY2017 + WY2023, each model at its best-validation checkpoint.
NSE/KGE are pooled over both windows; peak metrics are computed per window and
averaged, so the seam between disjoint windows cannot register as an event
(2026-08-17; the earlier concatenated values differed by ≤0.12 in Missed-Peaks).

| Run | h | best ep | test NSE | KGE | FHV | Missed-Pk | Peak-MAPE | focus-5 NSE |
|---|---|---|---|---|---|---|---|---|
| Determ., original split¹ | 16 | 15 | 0.836¹ | — | — | — | — | 0.841¹ |
| Determ., original split¹ | 128 | 13 | **0.862**¹ | — | — | — | — | 0.923¹ |
| Determ., flood split | 16 | 15 | 0.679 | 0.765 | −12.2% | 0.333 | 48.1% | 0.816 |
| CMAL, flood split | 16 | 2 | 0.517 | 0.411 | −60.4% | 0.667 | 70.1% | 0.425 |
| Determ., flood split | 128 | 14 | 0.754 | **0.776** | −18.6% | 0.450 | 47.6% | 0.819 |
| **CMAL, flood split** | **128** | **16** | **0.784** | 0.729 | −20.9% | **0.333** | **42.4%** | **0.824** |

¹ 2012–2014 drought window, same 22-basin cohort (that window has 26 scoreable basins;
the h16 median over all 26 is 0.808). The h128 original-split run (config
`ca-28basin-orig-h128-config.yml`, best-val epoch 13, added 2026-08-17) makes the
headline drought→flood comparison same-capacity: **0.862 → 0.754**. Note the flood
split trains on ~11 *more* years, so the window effect is, if anything, understated.

**Two findings carry the project:**

- **NSE falls 0.862 → 0.754 at matched capacity (h128) from the drought window to real
  flood years**, on identical basins (h16: 0.836 → 0.679). Peak *timing* is good (~1 day) but magnitude is not (FHV ≈ −18%,
  ~47% of peaks missed): **the model gets *when*, not *how big*.**
- **Capacity buys average accuracy and costs peak accuracy.** Deterministic h16→h128
  improved NSE 0.679→0.754 but worsened FHV (−12.2%→−18.6%) and Missed-Peaks
  (0.333→0.450) — consistent with regression-to-the-mean on OOD extremes.
  Single-seed caveat: each configuration is one training run; LSTM results move
  ±0.01–0.03 NSE across seeds, so treat the capacity-vs-peaks gradient as
  consistent-with, not established-by, this experiment.

**The CMAL story was a capacity artifact, twice over.** At h=16 CMAL looked far worse
and looked like it was overfitting; it was **under-capacity** (12 output params per
timestep vs 1, with settings copied from a `hidden_size: 512` reference config). At
h=128 CMAL **beats the deterministic model on NSE (0.784 vs 0.754), missed peaks, and
peak-magnitude error, at the cost of KGE** — and it provides predictive intervals the
deterministic model cannot. Its metric-level bootstrap std is ≤0.001 NSE
(`make_cmal_point_metrics.py`), so the differences are not ensemble-sampling
noise; both models are single training runs, so seed-level variance is not excluded.

### 4.1b Ungauged transfer — leave-one-basin-out (2026-08-13)

Each focus basin fully held out of training; model trained on the other 27
(`configs/lobo/`, h=128, 15 epochs). Tested on WY2017 + WY2023.

| Basin | NSE | KGE | FHV | Peak-MAPE | Note |
|---|---|---|---|---|---|
| Merced @ Happy Isles | 0.857 | 0.713 | −15.2% | 49.1% | ⚠ nested |
| Merced @ Pohono | 0.790 | 0.657 | −16.3% | 62.3% | ⚠ nested |
| Bear Ck | 0.646 | 0.462 | −29.3% | 56.5% | clean |
| Pitman Ck | 0.375 | 0.214 | −61.6% | 80.3% | clean |
| **Mill Ck (rain-driven)** | **−0.740** | 0.022 | **+80.3%** | 55.6% | clean |

**⚠ The Merced pair is not a valid ungauged test.** Happy Isles is upstream of Pohono on
the same river (the spec's own "nested-catchment scaling case"), so holding one out leaves
its nested partner in training and leaks the hydrograph. **Do not quote 0.857/0.790 as
ungauged skill.** The genuinely independent tests are Bear Ck, Pitman Ck and Mill Ck —
**median NSE 0.375**, far below the 0.754 gauged flood-year performance.

**The headline finding: the model regionalizes within a hydrologic regime and fails across
regimes.** Mill Ck is the only rain-driven basin among the five. Held out from 27
mostly-snowmelt basins it does not merely degrade — it inverts, scoring NSE −0.740 with
FHV **+80%**, over-predicting high flow where every other basin under-predicts. That is the
operationally important caveat for ungauged prediction, and it is a far sharper claim than
"ungauged transfer works."

### 4.1c CMAL evaluation spread — SUPERSEDED 2026-08-14

The 2026-08-13 "10 repeated evaluations" study (NSE 0.769 ± 0.027, KGE 0.690 ± 0.029,
FHV std 11.4) attributed run-to-run scatter to unseeded sampling. **That diagnosis was
wrong.** The scatter was the framework's CMAL evaluation path scoring against corrupted
observations that differ per run (trap §5.2, revised). Scored correctly — sample-median
point summary against source-netCDF obs — the CMAL metrics are essentially deterministic:
bootstrap std over re-drawn 7500-sample ensembles is ≤0.001 NSE, ≤0.2 Peak-MAPE
(`make_cmal_point_metrics.py`, `cmal_point_metrics.csv`). The old KGE 0.690 sat 2.7σ
from a clean evaluation's 0.729–0.770 — contamination, not noise. Use the §4.1 table.

### 4.1d Probabilistic scoring (2026-08-13) — `make_probabilistic_scores.py`

The project's stated differentiator. (An earlier draft argued the two heads were
statistically tied on point metrics; that reading traced to the §5.2 scoring bug —
§4.1 now shows CMAL ahead on NSE and peaks. The distributional comparison stands on
its own: point metrics collapse 7500 samples to one number.)

**Why the comparison is fair: CRPS generalizes MAE.** For a deterministic forecast — a
point mass — CRPS collapses exactly to `|obs − pred|`. So both models are scored on the
*same* proper scoring rule with no special-casing, and the ensemble only wins if its
spread is genuinely informative. Fair (finite-ensemble-unbiased) CRPS is used, the same
variant WeatherNext optimises.

**Rescored 2026-08-14** on the regenerated clean zarrs, observations from the verified
deterministic store, and both models on the identical valid-day mask (the first pass
scored the two models on different day counts and, for a handful of basins, against
corrupted observations — trap §5.2).

**Result 1 — CMAL wins decisively on CRPS, at every lead.** Median over the cohort:

| Lead (d) | CMAL | Deterministic | CMAL improvement |
|---|---|---|---|
| 0 (nowcast) | 0.916 | 1.231 | **25.6%** |
| 1 | 1.070 | 1.378 | 22.4% |
| 3 | 1.081 | 1.452 | 25.5% |
| 5 | 1.121 | 1.461 | 23.3% |
| 7 | 1.185 | 1.550 | 23.6% |

CRPS is in mm/day. Lead = zarr time_step (0 = same-day nowcast); an earlier CSV
labeled these off by one (fixed 2026-08-17).

A consistent **22–26% CRPS reduction**. This is the metric that matters for a forecast
product, and beyond §4.1's NSE edge it is where the distributional head separates.

**Result 2 — but the intervals are NOT calibrated.** Nominal 90% predictive interval:

| Lead (d) | Coverage | Verdict | Spread/skill | Reliability (TV) |
|---|---|---|---|---|
| 0 (nowcast) | 0.742 | fails | 3.27 | 0.276 |
| 3 | 0.705 | fails | 3.03 | 0.283 |
| 7 | 0.664 | fails | 2.71 | 0.292 |

Coverage of **0.66–0.74 against a nominal 0.90**, outside the pre-specified 85–95%
band at every lead. Per focus basin at lead 1: Mill Ck 0.805, Bear Ck 0.744, Merced HI
0.705, Merced Pohono 0.622, **Pitman Ck 0.537**.

**Result 3 — the diagnosis, settled by the rank histogram (2026-08-17,
`make_pit_diagnostic.py`).** Coverage says the intervals miss 26–34% of observations,
while spread/skill ≈ 3 says the ensemble standard deviation is ~3× the RMSE. An earlier
draft attributed both to heavy tails alone; review noted conditional bias produces the
same two symptoms, so the PIT/rank histogram was computed (n = 15,907 basin-days,
nowcast). It is U-shaped and nearly symmetric in aggregate (22% of PIT > 0.9, 20% <
0.1) — but conditioning on flow reveals **opposite-signed misses: at high flows the
observation escapes above the interval (22.7% above vs 3.0% below); at low flows it
escapes below (22.8% vs 8.6%)**. The interval is not merely too narrow — it is
mis-centered toward the middle of the flow distribution at both ends: regression toward
the mean, expressed distributionally. The far-tail samples that inflate spread/skill
(and cause the NaN losses) are a separate, genuine heavy-tail pathology.
**Consequence for remediation: uniform post-hoc variance inflation cannot fix this** —
the high-flow misses are displacement, not dispersion. The remedy is flow-conditional
recalibration (or a bias-aware head), and any fix must be checked against the CRPS gain.

**This root cause explains the genuinely tail-driven CMAL pathologies:**
- the NaN training losses (a mixture component's scale blowing up),
- the `mean` sample-reduction catastrophe (NSE −84, FHV +284% — extreme samples destroying
  the mean while the median stayed robust),
- the `_sample_asymmetric_laplacians` shape error, and the ~2×10⁻⁶ of samples
  exceeding 10⁴ mm/day in an otherwise-clean ensemble.

(The §4.1c evaluation scatter, previously on this list, turned out to be the separate
obs-corruption bug — trap §5.2 — not heavy tails.)

**Honest verdict: the distributional head earns its keep on sharpness and fails on
calibration.** For an operational product that is a real problem — the forecast is
better *and* overconfident about being better. Remediation to try: fewer mixture
components (`n_distributions` 3 → 1–2), or the GMM head, which the framework's own
docstring describes as less brittle than CMAL.

### 4.1e Held-out January-1997 protocol (recorded 2026-08-17; experiment run 2026-06-06)

The 1997 claims quoted in the writeup come from `make_1997.py` and config
`ca-28basin-1997-config.yml`: h=16 deterministic head, **train 2002–2014, validation
2000–2001, test 1993–1999** (the entire 1990s excluded from training), scored on water
year 1997 against the NWM **v2.1** retrospective at the focus gauges. Forcing caveat: in
1997 the forecast-era products (HRES, IMERG) do not exist; `union_mapping` backfills
from ERA5-Land/CPC, so the model runs on degraded inputs relative to the 2017/2023 test
years — part of the peak underestimate may be forcing, not model. Peak errors: LSTM −43%
to −78%, NWM −7% to −86%; both time the peak to ~1 day.

### 4.2 Honest baselines — `make_baselines.py`

**Recomputed 2026-08-13 on the flood window (WY2017 + WY2023), climatology fitted on the
flood-split TRAINING periods only.** These are apples-to-apples with §4.1.

**LSTM reference (deterministic h128 @ep14): cohort median NSE 0.754, focus-5 median
0.819** (per-lead values below from `make_lstm_by_lead.py`, `lstm_by_lead.csv`).

| Baseline | 22-basin cohort | focus-5 |
|---|---|---|
| train-period mean | −0.084 | −0.203 |
| day-of-year climatology | **+0.198** | +0.218 |
| persistence (lead 1) | +0.712 | **+0.944** |
| damped persistence | +0.738 | +0.945 |

**Climatology flipped sign.** It scored **−0.25** on the old 2012–2014 window and **+0.198**
here. The negative result was an artifact of that window being drought onset — a
1985–2008 climatology systematically over-predicts a drought. On genuine flood years
climatology is a real, positive baseline. *The lesson generalizes: a baseline's value is a
property of the test period, not of the baseline.*

**Persistence vs the LSTM, matched per lead (CORRECTED 2026-08-14).** The earlier
version of this comparison quoted the LSTM as a single flat number against per-lead
persistence. That pairing was wrong: `time_step` k in the results zarr is the k-day-ahead
forecast issued on day d (tester.py stamps `date` = issue date), so the fair pairing is
time_step k ↔ persistence lead k. time_step 0 is a same-day nowcast (the model has
day-d weather analysis but never any gauge) and has no exact persistence analogue;
its nearest comparison is lead-1 persistence.

| Lead (d) | LSTM cohort | Pers. cohort | LSTM focus-5 | Pers. focus-5 | LSTM Mill Ck | Pers. Mill Ck |
|---|---|---|---|---|---|---|
| 0 (nowcast) | 0.754 | — | 0.819 | — | 0.726 | — |
| 1 | 0.667 | **0.712** | 0.810 | **0.944** | **0.618** | 0.541 |
| 2 | **0.688** | 0.484 | 0.810 | **0.864** | **0.619** | 0.157 |
| 3 | **0.665** | 0.301 | **0.809** | 0.803 | **0.606** | −0.059 |
| 5 | **0.670** | 0.037 | **0.802** | 0.702 | **0.666** | −0.234 |
| 7 | **0.597** | −0.122 | **0.792** | 0.610 | **0.512** | −0.516 |

**The defensible claim, correctly paired:**

- **At 1 day ahead, yesterday's gauge beats the LSTM nearly everywhere** — cohort 0.712
  vs 0.667, focus-5 0.944 vs 0.810. The earlier "LSTM beats persistence at every lead
  across all basins" does not survive the pairing correction.
- **The LSTM overtakes at 2 days ahead across the cohort** (0.688 vs 0.484 plain,
  0.566 damped) **and on the high-storage snowmelt focus basins at 3 days vs plain
  persistence (0.809 vs 0.803) but only at 4 days vs damped persistence** (0.811 vs
  0.768; damped still wins day 3, 0.814 vs 0.809). Damped persistence
  (climatology + α^k × lag-k anomaly, α fit on training) is the stronger null and
  decays slower; per-lead values are in `persistence_by_lead.csv` (damped_* columns,
  added 2026-08-17). Both nulls collapse at long leads while the LSTM decays slowly.
- **On rain-driven Mill Ck the LSTM wins at every lead** (0.618 vs 0.541 at lead 1;
  persistence is negative by lead 3) — flashy rain response is not persistent.

**The asymmetry must travel with these numbers:** the LSTM is forcing-driven and **never
ingests observed discharge** (`hindcast_inputs` are HRES/IMERG/CPC precipitation and
temperature only). Persistence uses yesterday's gauge reading, which the model is denied —
but a real operator *does* have the gauge. Reframed claim: *beats NWM, and beats
gauge-persistence from ~day 3–4 on snowmelt basins and at every lead on rain-driven ones.*
This points directly at the obvious next architecture: **assimilate the gauge.**

### 4.3 NWM benchmark

**Original result (2012–2014 drought window, NWM v2.1):** LSTM median NSE **0.83** vs
**0.53**, winning all five focus basins. Model-vs-model on identical data.

### 4.3b NWM benchmark on the flood years (2026-08-14) — the win does not generalize

`make_benchmark_flood.py`. NWM v2.1 retrospective ends 2020-12-31 and v3.0 ends
2023-02-01, so: **WY2017** compares LSTM vs v2.1 vs v3.0 (full year); **WY2023p**
compares LSTM vs v3.0 on 2022-10-01→2023-01-31 only (includes the late-Dec–mid-Jan AR
sequence, misses the March 2023 events — state this wherever the number appears).
Windows scored separately; both models against the identical observed series; NWM
converted m³/s → mm/day via Caravan basin areas (≤4.5% error, §2.2); the retrospective
is analysis-forced, so it is compared to the LSTM's lead-0 hindcast. Focus-basin
medians (per-basin table written by `make_benchmark_flood.py`):

| Window | Model | NSE | FHV | Peak-MAPE |
|---|---|---|---|---|
| WY2017 | LSTM | 0.772 | −19.8% | 48.0% |
| WY2017 | NWM v2.1 | 0.628 | −2.4% | 32.1% |
| WY2017 | **NWM v3.0** | **0.848** | **−4.6%** | **35.2%** |
| WY2023p | **LSTM** | **0.443** | −53.0% | 90.7% |
| WY2023p | NWM v3.0 | 0.162 | −54.9% | 82.8% |

**What survives and what does not:**

- The 2012–2014 "beats NWM on every basin" **does not extend to flood years against the
  modern NWM.** On WY2017, **NWM v3.0 beats the LSTM on median NSE and is far better on
  high-flow bias and peak error**; v2.1 remains worse on NSE, so a large share of the
  original margin was the NWM *version*, and the benchmark must always be labeled v2.1.
- **The LSTM wins WY2023p on average skill** (0.443 vs 0.162) but both systems are poor
  there, with severe high-flow underestimation (FHV ≈ −53/−55%) — on the hardest AR
  window nobody wins.
- **Bear Ck is the LSTM's structural win in both windows** (0.924 vs −0.18/0.07 in
  WY2017; 0.767 vs −0.45 in WY2023p) — the documented NWM snowmelt failure at
  high-storage basins. **Rain-driven peaks remain NWM's structural win** (Mill Ck
  WY2023p 0.896 vs 0.737; consistent with the held-out-1997 result).
- Peak-count metrics (Missed-Peaks, Peak-Timing) are computed from few events per
  window; treat NSE/FHV/Peak-MAPE as primary here.

---

## 5. Traps — read before trusting any number

### 5.1 CMAL run-to-run scatter — RETIRED 2026-08-14

Repeated `run evaluate` calls on the same CMAL checkpoint disagreed (NSE 0.728–0.768,
FHV std 11.4), which was attributed to unseeded sampling. Wrong: with 7500 samples the
sample-median's true sampling noise is ≤0.001 NSE (bootstrap, `make_cmal_point_metrics.py`).
The scatter was trap §5.2. **Never score CMAL through the framework's evaluation path;
score its stored samples against source observations.**

### 5.2 The CMAL evaluation path corrupts observations — REVISED 2026-08-14

A CMAL `run infer` writes corrupted `streamflow_obs` into `test_results.zarr` for a
**nondeterministic subset of basins** — recurring garbage constants (1.966179 on
several basins' first test day), values at 10³⁰–10³⁸ and ±inf, and phantom "valid"
days on basins with no observations at all. **This reproduces on a clean, uncontended
GPU**, so the 2026-08-13 diagnosis of "MPS contention" was wrong for this failure; it
is an upstream bug in the sampling path's obs handling. The written *simulations* are
unaffected (sample medians remain coherent against source data). Deterministic runs are
unaffected: their zarr obs pass closure against the source netCDFs bit-for-bit, and
repeated inference is bit-reproducible (median NSE 0.754310 across three runs).

Consequences: (a) obs must come from the source netCDFs or a deterministic run's
verified store, never from a CMAL zarr; (b) any metric the framework logs for a CMAL
run — including its in-memory `run evaluate` output — is suspect, because the same
buffer feeds the metric computation.

**Diagnostic heuristic (unchanged and vindicated):** when metrics are *internally
contradictory* — NSE 0.80 alongside FHV −100% cannot both be true — suspect the
measurement, not the model. Sanity-check the **observations**, not just the simulations.

### 5.3 One run directory per attempt

A retry loop creates a **new timestamped run directory per attempt**. Globbing
`runs/<name>_*/output.log` concatenates several unrelated validation curves into a
meaningless trajectory. Always read a single run directory.

### 5.4 CMAL is numerically brittle

Three runs died with `RuntimeError: Loss was NaN for 1 times in a row` after 6, 3 and 15
epochs, and one hit a tensor-shape error in `_sample_asymmetric_laplacians` at epoch 25.
This is documented in the framework's own CMAL docstring ("more brittle than GMM").
Mitigation: `allow_subsequent_nan_losses: 10` — the framework *skips* NaN steps and only
aborts after N consecutive ones, and the default of 0 means a single NaN kills the run.
Chosen deliberately over lowering the learning rate, which would have changed training
dynamics and broken comparability. **It tolerates NaNs; it does not prevent them.** NaN
occurrence is nondeterministic on MPS even with `seed: 42` fixed.

### 5.5 Laptop sleep is survivable; it just costs wall clock

A sleep event leaves an unmistakable gap in the epoch timestamps (observed: 88 minutes
between epochs 17 and 18) and training resumed cleanly with MPS state intact. Useful as a
control: failures *without* a timestamp gap are not sleep-related.

### 5.6 dtype must be preserved when writing extended netCDFs

`reindex`/assignment upcasts float32 → float64, and the data loader hard-asserts float32
(`Data variable or coord 'streamflow' is a float but not float32`). Cost a full training
launch before it was caught.

### 5.7 USGS NWIS throttles silently

Rapid repeated queries return **empty responses that look like "no data" rather than
errors**. An early closure-test run excluded 11 basins including 4 of 5 focus basins for
this reason alone. Fixed with an on-disk cache (`~/data/nwis_cache`) plus retry, so the
fit and the closure test read identical bytes.

---

### 5.8 `run evaluate` reports a silently-restricted basin cohort — 2026-08-14

`evaluate` sets `tester_skip_obs_all_nan: true` (from the config) and its
`_calc_exclude_basins` drops **whole basins** from the metrics CSV — on the flood split
it scored **16 of 28 basins**, excluding 6 that have valid flood-window data. `infer`
scores all 28 rows (22 with data). Per-basin metrics are **bit-identical** between the
two verbs; only the cohort differs. This is where the long-standing 0.746-vs-0.754
median discrepancy came from — neither number was corrupted; they are medians over
different basin sets. **Always take metrics from the infer path and state the cohort
(22 of 28).**

### 5.9 `run evaluate` DELETES `test_results.zarr` — 2026-08-14

Both verbs clear the epoch's test directory before writing. Running `run evaluate` to
"just check the metrics" destroys the zarr an earlier `run infer` produced (it cost two
regenerations on 2026-08-14). Copy metrics CSVs aside before re-running either verb.

---

## 6. State and next steps

**2026-08-14: the numbers are final and internally consistent.** All headline results
now flow from four scripts under one protocol (`make_model_comparison.py`,
`make_lstm_by_lead.py` + `make_baselines.py`, `make_probabilistic_scores.py`,
`make_benchmark_flood.py`) with obs from source netCDFs and the 22-basin cohort stated.
The remaining work is presentation: figures for the flood-split story, the public
writeup refresh (NWM flood-year section, corrected persistence pairing, corrected
cohort medians), and publication of this repo.

**Explicitly deferred** (real science, does not change the writeup's conclusions):
peak attribution (forcing/loss/OOD/capacity arms), calibrated SAC-SMA/MARRMoT baseline,
CMAL calibration remediation (fewer components / GMM head / post-hoc variance scaling),
gauge assimilation (the architecture the persistence result points at).
