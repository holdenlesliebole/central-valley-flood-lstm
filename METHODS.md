# Central Valley flood-forecasting LSTM — methods, results, and traps

**Last updated 2026-08-13.** This is the reproducibility and methodology record for the
code in this repo. The portfolio-facing summary lives in
`~/Documents/Job_Search/portfolio/central_valley_hydrology.md`; results here and there
must not drift.

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

### 4.1 Model comparison

Flood-split test = WY2017 + WY2023, each model at its best-validation checkpoint.

| Run | h | best ep | val NSE | test NSE | KGE | FHV | Missed-Pk | Peak-MAPE |
|---|---|---|---|---|---|---|---|---|
| Determ., original split | 16 | 15 | 0.836 | **0.808**¹ | 0.786 | — | — | — |
| Determ., flood split | 16 | 15 | 0.823 | 0.671 | 0.754 | −17.4% | 0.367 | 53.2% |
| CMAL, flood split | 16 | 2 | 0.612 | 0.463 | 0.416 | −66.3% | 0.974 | 72.4% |
| **Determ., flood split** | **128** | **14** | 0.870 | **0.746** | **0.744** | −22.4% | 0.500 | 50.4% |
| **CMAL, flood split** | **128** | **16** | **0.892** | ~0.745² | 0.653 | ~−21%² | ~0.48² | **~42%²** |

¹ on the 2012–2014 drought window — not comparable to the flood-split column.
² mean of 3 repeated evaluations; see §5.1 — single CMAL runs are not reproducible.

**Two findings carry the project:**

- **NSE falls 0.808 → 0.746 from the drought window to real flood years.** The original
  headline was flattered by a benign test period. Peak *timing* is good (~1 day) but
  magnitude is not (FHV ≈ −20%, ~50% of peaks missed): **the model gets *when*, not
  *how big*.**
- **Capacity buys average accuracy and costs peak accuracy.** Deterministic h16→h128
  improved NSE 0.671→0.746 but worsened FHV (−17.4%→−22.4%) and Missed-Peaks
  (0.367→0.500) — consistent with regression-to-the-mean on OOD extremes, and a fourth
  arm for the peak-attribution experiment using runs already on disk.

**The CMAL story was a capacity artifact, twice over.** At h=16 CMAL looked far worse and
looked like it was overfitting; it was **under-capacity**. The head must parameterize 12
numbers per timestep (3 components × 4 params) versus 1 for regression, and the settings
were copied from a reference config using `hidden_size: 512`. Raised to h=128, CMAL's
validation NSE went 0.612 → 0.892 and it now **beats** the deterministic model on
validation NSE, peak magnitude error, and peak timing, while matching it on NSE, FHV and
missed peaks — and it provides predictive intervals the deterministic model cannot.

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
**median NSE 0.375**, far below the 0.746 gauged flood-year performance.

**The headline finding: the model regionalizes within a hydrologic regime and fails across
regimes.** Mill Ck is the only rain-driven basin among the five. Held out from 27
mostly-snowmelt basins it does not merely degrade — it inverts, scoring NSE −0.740 with
FHV **+80%**, over-predicting high flow where every other basin under-predicts. That is the
operationally important caveat for ungauged prediction, and it is a far sharper claim than
"ungauged transfer works."

### 4.1c CMAL Monte Carlo spread — 10 repeated evaluations (2026-08-13)

Same checkpoint (ep16), same config, ten runs:

| Metric | mean | std | min | max | vs determ. h128 |
|---|---|---|---|---|---|
| NSE | 0.769 | 0.027 | 0.740 | 0.819 | CMAL ahead <1σ (marginal) |
| KGE | 0.690 | 0.029 | 0.652 | 0.738 | **deterministic better (~1.8σ)** |
| Peak-MAPE | 47.0 | 2.03 | 44.3 | 50.4 | **CMAL better (~1.7σ)** |
| FHV | −22.9 | **11.43** | −55.1 | −17.6 | indistinguishable |
| Missed-Peaks | 0.549 | 0.152 | 0.429 | 0.894 | indistinguishable |
| Peak-Timing | 0.887 | 0.124 | 0.750 | 1.000 | indistinguishable |

**Honest verdict: CMAL is better on peak magnitude error, worse on KGE, and
indistinguishable elsewhere.** Not a sweep. **FHV's std of 11.4 across identical runs makes
it unusable for CMAL from a single evaluation** — which retroactively explains the −38.6%
and −100% single-run readings that briefly looked like model failure.

### 4.1d Probabilistic scoring (2026-08-13) — `make_probabilistic_scores.py`

The project's stated differentiator, finally built. Necessary rather than optional: after
§4.1c the two models are statistically tied on point metrics, so this is the only thing
that can separate them.

**Why the comparison is fair: CRPS generalizes MAE.** For a deterministic forecast — a
point mass — CRPS collapses exactly to `|obs − pred|`. So both models are scored on the
*same* proper scoring rule with no special-casing, and the ensemble only wins if its
spread is genuinely informative. Fair (finite-ensemble-unbiased) CRPS is used, the same
variant WeatherNext optimises.

**Result 1 — CMAL wins decisively on CRPS, at every lead.** Median over 28 basins:

| Lead (d) | CMAL | Deterministic | CMAL improvement |
|---|---|---|---|
| 1 | 0.945 | 1.231 | **23.3%** |
| 2 | 1.072 | 1.378 | 22.2% |
| 4 | 1.080 | 1.452 | **25.6%** |
| 6 | 1.121 | 1.461 | 23.3% |
| 8 | 1.185 | 1.550 | 23.6% |

A consistent **22–26% CRPS reduction**. This is the first metric that clearly separates
the two models, and it is the one that matters for a forecast product.

**Result 2 — but the intervals are NOT calibrated.** Nominal 90% predictive interval:

| Lead (d) | Coverage | Verdict | Spread/skill | Reliability (TV) |
|---|---|---|---|---|
| 1 | 0.691 | fails | 2.89 | 0.302 |
| 4 | 0.660 | fails | 2.90 | 0.308 |
| 8 | 0.639 | fails | 2.84 | 0.326 |

Coverage of **0.64–0.69 against a nominal 0.90**, far outside the pre-registered 85–95%
band. Per focus basin at lead 1: Mill Ck 0.805, Bear Ck 0.752, Merced HI 0.704, Merced
Pohono 0.623, **Pitman Ck 0.536**.

**Result 3 — the diagnosis, from an apparent contradiction.** Coverage says the intervals
are too *narrow* (under-dispersed), while spread/skill ≈ 2.9 says the ensemble standard
deviation is nearly 3× too *large*. Both are true only if the predictive distribution is
**heavy-tailed**: variance inflated by rare extreme samples, while the central 90% interval
remains too concentrated. Too little mass in the "moderately wrong" range, too much in the
far tail.

**This single root cause explains every CMAL pathology observed in this project:**
- the NaN training losses (a mixture component's scale blowing up),
- FHV's std of 11.4 across identical evaluations (§4.1c),
- the `mean` sample-reduction catastrophe (NSE −84, FHV +284% — extreme samples destroying
  the mean while the median stayed robust),
- the `_sample_asymmetric_laplacians` shape error.

**Honest verdict: the distributional head earns its keep on sharpness and fails on
calibration.** For an operational product that is a real problem — the forecast is
better *and* overconfident about being better. Remediation to try: fewer mixture
components (`n_distributions` 3 → 1–2), or the GMM head, which the framework's own
docstring describes as less brittle than CMAL.

### 4.2 Honest baselines — `make_baselines.py`

**Recomputed 2026-08-13 on the flood window (WY2017 + WY2023), climatology fitted on the
flood-split TRAINING periods only.** These are apples-to-apples with §4.1.

**LSTM reference (deterministic h128 @ep14): all-28 median NSE 0.746, focus-5 median
0.819.**

| Baseline | all-28 | focus-5 |
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

**Persistence vs lead time — the binding comparison:**

| Lead (d) | Persistence all-28 | Persistence focus-5 |
|---|---|---|
| 1 | +0.712 | **+0.944** |
| 2 | +0.484 | **+0.864** |
| 3 | +0.301 | +0.803 |
| 4 | +0.133 | +0.752 |
| 5 | +0.037 | +0.702 |
| 7 | −0.122 | +0.610 |

**The defensible claim, on the correct test set:**

- **Across all 28 basins the LSTM (0.746) beats gauge-persistence at every lead**, though
  only narrowly at lead 1 (0.712).
- **On the five high-storage snowmelt focus basins, persistence wins at leads 1–2
  (0.944, 0.864 vs the LSTM's 0.819) and the LSTM overtakes at day 3** (0.803).
- **On rain-driven Mill Ck the LSTM wins at every lead** — 0.726 vs persistence 0.541 at
  lead 1 — because flashy rain response is not persistent.

**The asymmetry must travel with these numbers:** the LSTM is forcing-driven and **never
ingests observed discharge** (`hindcast_inputs` are HRES/IMERG/CPC precipitation and
temperature only). Persistence uses yesterday's gauge reading, which the model is denied —
but a real operator *does* have the gauge. Reframed claim: *beats NWM, and beats
gauge-persistence from ~day 3–4 on snowmelt basins and at every lead on rain-driven ones.*
This points directly at the obvious next architecture: **assimilate the gauge.**

### 4.3 NWM benchmark (2012–2014, unaffected by the above)

LSTM median NSE **0.83** vs NWM v2.1 **0.53**, winning all five focus basins. Model-vs-model
on identical data, so the baseline and split findings do not touch it.

---

## 5. Traps — read before trusting any number

### 5.1 CMAL evaluation is NOT reproducible

CMAL draws `n_samples: 7500` **unseeded** stochastic samples, so metrics vary run to run.
Three identical repeats of the same checkpoint:

| Repeat | NSE | FHV | Missed-Peaks |
|---|---|---|---|
| 1 | 0.728 | −22.8% | 0.586 |
| 2 | 0.768 | −18.4% | 0.429 |
| 3 | 0.740 | −22.0% | 0.429 |

**Always report CMAL as a mean over repeats with the spread stated.** The deterministic
model has no sampling step and is exactly reproducible.

### 5.2 NEVER run inference while training occupies the GPU

MPS contention **silently corrupts tensors rather than erroring**. A `run infer` executed
during training produced a `test_results.zarr` whose `streamflow_obs` contained values from
−9.2×10³⁷ to +2.5×10³¹ while the source netCDF was clean (0.04–76.2 mm/day). The same
contention produced an impossible FHV of −100% that was briefly mistaken for model failure.

**Diagnostic heuristic:** when metrics are *internally contradictory* — NSE 0.80 alongside
FHV −100% cannot both be true — suspect the measurement, not the model. Sanity-check the
**observations**, not just the simulations.

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

## 6. State and next steps

**Running (2026-08-13 overnight):** 10 repeated CMAL evaluations to quantify §5.1 spread
properly, then leave-one-basin-out training for the 5 focus basins (`configs/lobo/`,
15 epochs each, independently checkpointed).

**Next:**
1. **CRPS / coverage / reliability scoring** on the clean zarr — the project's own stated
   differentiator, still unbuilt, and the only fair comparison for a distributional model
   (point metrics collapse 7500 samples to one number).
2. **Recompute baselines on WY2017/WY2023** (§4.2 warning).
3. **Peak attribution**: forcing-product arm, loss-geometry arm (runs already on disk),
   OOD-distance arm, plus the new capacity arm.
4. Calibrated process baseline — SAC-SMA + SNOW-17, or the MARRMoT suite.
5. Rewrite the public writeup, which still carries the superseded 2026-06 framing.
