# Tier-1 remediation memo — adversarial review of 2026-08-19

**Status: APPLIED 2026-08-19.** The prose changes listed in §8 have been made in
`docs/METHODS.md`, `paper/main.tex`, the public writeup, and the site note; this memo is
the record of what changed and why.

**Date:** 2026-08-19
**Scope:** the two mechanical scoring bugs the review called stop-ship (its finding 1), plus the
two quantifications it asked for before any claim set is frozen (its findings 2 and 4).
**Not in scope:** retraining, the forecast-provenance audit (review finding 3), the
development-set reclassification (finding 5), and every item in the review's "major scientific
limitations" section. Those remain open.

---

## 1. What was wrong, in one sentence

The zarr's `date` coordinate is the forecast **issue** date and `time_step = k` is the **lead**,
so row `(d, k)` predicts flow on the **valid** date `d + k` — and two analysis scripts read
`date` as the valid date, scoring every lead against observations `k` days too early.

The convention is not a matter of interpretation. `googlehydrology/evaluation/tester.py` sets
`date_coords = dates[lowest_freq][:, -lead_time - 1]` and `time_step_coords += lead_time`;
`datasetzoo/multimet.py::_calc_date_range(..., lead=True)` extracts the target sequence ending at
`date + lead_time`; and `predict_last_n: 8 = lead_time + 1` in the config means a sample's labels
are exactly the eight dates `d .. d+7`. I read those lines rather than taking the review's word
for them, and then closure-tested the result against the source netCDFs (§2).

Lead 0 is unaffected by the bug, because `d + 0 = d`. Every headline lead-0 number in the project
— the 0.754 median NSE, the §4.1 model-comparison table, the §4.4 lead-0 storm bins, the
`make_probabilistic_scores.py` raw multi-lead results, `make_lstm_by_lead.py`, the NWM benchmark —
is untouched by this memo. `make_probabilistic_scores.py` was already correct: it reads
lead-specific observations out of the verified deterministic zarr (`verified_obs[basin][lead]`),
which the closure test below shows is the right pairing.

---

## 2. Closure test (new: `scripts/test_valid_date_alignment.py`)

The test asserts two things against the deterministic h128 flood zarr
(`runs/ca-28basin-flood-h128_1208_025528/test/model_epoch014/test_results.zarr`), scoring against
the source Caravan netCDFs, never a CMAL zarr:

1. **aligned** — `zarr_obs(d, k) == source_q(d + k)` within float32 tolerance, for all `k = 0..7`;
2. **not vacuous** — `zarr_obs(d, k)` vs `source_q(d)` is grossly wrong for every `k >= 1`, so the
   test could actually have failed.

Result (15,907 finite basin-day pairs per lead, 28 basins, 2,556 issue dates):

| lead | n pairs | max abs err, aligned (mm/day) | max abs err, unaligned (mm/day) |
|---:|---:|---:|---:|
| 0 | 15907 | 7.62939e-06 | 7.62939e-06 |
| 1 | 15907 | 7.62939e-06 | 112.866 |
| 2 | 15907 | 7.62939e-06 | 133.595 |
| 3 | 15907 | 7.62939e-06 | 129.040 |
| 4 | 15907 | 7.62939e-06 | 129.741 |
| 5 | 15907 | 7.62939e-06 | 131.201 |
| 6 | 15907 | 7.62939e-06 | 135.597 |
| 7 | 15907 | 7.62939e-06 | 134.190 |

**Reproduces the review exactly**: it reported max error 7.63e-6 mm/day aligned and 113–136
mm/day unaligned.

Magnitude hand-check (guardrail Rule 1): 7.62939e-06 is exactly 2⁻¹⁷, which is one float32 ULP at
a value near 2⁶ = 64 mm/day. So the "error" is a single-bit float32 representation difference on
the largest flows in the record — the observations are bit-identical, not merely close. The
unaligned column is a *k*-day displacement of a flashy daily hydrograph, and 113–136 mm/day is the
size of the biggest storm days in the record. Both columns are what they should be.

Lead 0 having identical aligned and unaligned errors is the invariant that proves the test is
wired correctly: at lead 0 the two pairings are the same pairing.

---

## 3. Fix 1 — `scripts/make_recalibration.py`

**Change.** `gather()` now builds `valid_date = issue_date + lead` per lead, reads source
observations at the valid date, and returns both date vectors alongside every row; nothing
downstream can construct a row without both. The cross-fitting windows (WY2017 / WY2023) are
assigned on the **valid** date. Rows whose valid date falls outside both test water years — the
lead overhang past each window end, whose targets are training labels — are dropped (1,232 of
127,256 rows, 0.97%) and the count is printed. Outputs gain `issue_date_first/last` and
`valid_date_first/last` columns, plus a new long-form companion
`outputs/figures/recalibration_by_basin.csv` (basin × lead × variant, 528 rows, each stamped with
both date spans).

**Regenerated:** `outputs/figures/recalibration.csv`, `recalibration_by_basin.csv`,
`recalibration.png`.

### Old vs new, coverage of the nominal 90% interval

| lead | raw OLD | raw NEW | global OLD | global NEW | flow-cond OLD | flow-cond NEW |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.723 | 0.723 | 0.929 | 0.912 | 0.925 | 0.906 |
| 1 | 0.730 | 0.701 | 0.937 | 0.900 | 0.932 | 0.893 |
| 2 | 0.669 | 0.697 | 0.926 | 0.898 | 0.925 | 0.892 |
| 3 | 0.613 | 0.696 | 0.907 | 0.896 | 0.908 | 0.890 |
| 4 | 0.570 | 0.691 | 0.892 | 0.899 | 0.895 | 0.891 |
| 5 | 0.525 | 0.682 | 0.873 | 0.895 | 0.875 | 0.886 |
| 6 | 0.498 | 0.675 | 0.858 | 0.893 | 0.860 | 0.881 |
| 7 | 0.462 | 0.658 | 0.840 | 0.887 | 0.840 | 0.876 |

### Old vs new, fair CRPS (mm/day)

| lead | raw OLD | raw NEW | global OLD | global NEW | flow-cond OLD | flow-cond NEW |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.901 | 0.901 | 0.987 | 0.939 | 1.003 | 0.900 |
| 1 | 0.799 | 1.037 | 0.882 | 1.059 | 0.902 | 1.027 |
| 2 | 1.144 | 1.049 | 1.113 | 1.071 | 1.122 | 1.040 |
| 3 | 1.475 | 1.059 | 1.405 | 1.087 | 1.368 | 1.054 |
| 4 | 1.679 | 1.077 | 1.588 | 1.096 | 1.534 | 1.068 |
| 5 | 1.883 | 1.118 | 1.776 | 1.134 | 1.697 | 1.108 |
| 6 | 2.040 | 1.152 | 1.903 | 1.163 | 1.824 | 1.141 |
| 7 | 2.186 | 1.209 | 2.022 | 1.209 | 1.942 | 1.196 |

### Top-tercile (high-flow) days

| variant | coverage OLD | coverage NEW | CRPS OLD | CRPS NEW | escape-above OLD | escape-above NEW |
|---|---:|---:|---:|---:|---:|---:|
| raw | 0.530 | 0.697 | 3.846 | 2.854 | 0.391 | 0.261 |
| global | 0.843 | 0.888 | 3.666 | 2.892 | 0.151 | 0.109 |
| flow-conditional | 0.859 | 0.854 | 3.503 | 2.784 | 0.131 | 0.120 |

### Fitted parameters

New (all bands reach 90% fit-window coverage, none hits the s = 8 cap):
flow-conditional **mu 1.004–1.132, s 0.961–3.867**; global **mu 1.024–1.062, s 1.680–2.289**.
METHODS §4.5 records the pre-fix values as mu 0.90–1.24 / s 1.55–5.88 (flow-conditional) and
mu 1.08–1.14 / s 1.91–2.65 (global); I did not re-run the buggy code to re-derive those, so treat
them as quoted from METHODS rather than recomputed.

### What this does to the story

- **Raw undercoverage survives, its lead dependence does not.** Raw coverage is 0.66–0.72, nearly
  flat across lead, not 0.46–0.73 collapsing with lead. The claimed collapse was the misalignment.
- **Raw CRPS degradation with lead is mostly an artifact.** 0.90 → 1.21 mm/day across leads 0–7,
  not 0.90 → 2.19. The old lead-1 value (0.799) was even *better* than lead 0, which should have
  been the tell: a 1-day forecast cannot beat a same-day nowcast.
- **The "recalibration improves CRPS at long leads" claim is dead.** The global map now makes CRPS
  *worse* at every lead (e.g. 0.901 → 0.939 at lead 0; equal at lead 7). The flow-conditional map
  is roughly CRPS-neutral (0.901 → 0.900 at lead 0; 1.209 → 1.196 at lead 7 — a 1% gain, not
  2.19 → 1.94). The old improvement was recalibration partially undoing a date error.
- **Coverage recovery survives and is now better behaved**: 0.876–0.912 across all leads, versus
  raw 0.658–0.723. It no longer sags at leads 6–7.
- **One reversal.** On top-tercile days the flow-conditional map used to beat the global map on
  both coverage and CRPS. It now beats it on CRPS (2.78 vs 2.89) but *loses* on coverage
  (0.854 vs 0.888). "Beating a single global correction on both" is no longer true.
- The method still is not conformal prediction; the review is right that it should be called
  **cross-fitted affine calibration**. That is a naming fix, independent of the date bug.

---

## 4. Fix 2 — `scripts/make_storm_stratified_skill.py`

**Change.** The basin-day table is now long in lead (one row per basin × issue date × lead) and
carries `issue_date` and `valid_date`. Observation, ERA5-Land precipitation percentile, and storm
class are all taken at the **valid** date; the water-year window mask is applied on the valid date.
The 22-basin cohort test is deliberately still evaluated on the lead-0 axis so the cohort is
identical to the one used everywhere else in the project — it still returns the same 22 basins and
the same 6 exclusions.

**Regenerated:** `outputs/figures/storm_stratified_skill.csv`, `storm_stratified_skill.png`.

### Lead 3, old vs new

| storm bin | n OLD | n NEW | bias OLD | bias NEW | underpred OLD | underpred NEW | rel MAE OLD | rel MAE NEW | pooled NSE OLD | pooled NSE NEW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dry | 10645 | 10554 | −0.038 | **+0.017** | 0.581 | 0.507 | 0.408 | 0.282 | 0.521 | 0.814 |
| wet ≤P50 | 2368 | 2343 | −0.057 | **+0.007** | 0.565 | 0.501 | 0.461 | 0.288 | 0.405 | 0.771 |
| P50–P80 | 1550 | 1534 | −0.010 | **−0.039** | 0.485 | 0.540 | 0.554 | 0.320 | 0.201 | 0.731 |
| P80–P95 | 931 | 931 | −0.144 | **−0.296** | 0.542 | 0.734 | 0.487 | 0.431 | 0.439 | 0.569 |
| >P95 | 413 | 413 | −0.532 | **−0.614** | 0.738 | 0.879 | 0.627 | 0.644 | 0.130 | 0.154 |

Lead 0 is unchanged in every cell (−0.015, −0.008, +0.004, −0.219, −0.518; n = 10645, 2368, 1550,
931, 413; total 15,907 — the same 15,907 the closure test counts).

**Direction check.** Correcting the pairing removes a 3-day displacement error, so lead-3 relative
MAE falls sharply in the common bins (0.41 → 0.28 dry, 0.55 → 0.32 in P50–P80) and pooled NSE
rises (0.52 → 0.81 dry). In the two storm bins it goes the other way: aligned to the actual storm
day, the model's underprediction of the peak is *starker*, not milder (−0.144 → −0.296,
−0.532 → −0.614). That is the expected sign in both regimes, and it is why the review's summary —
"the direction survives, the numbers do not" — is the right verdict. The tail failure is worse
than published, not better.

**Independent confirmation (guardrail Rule 3).** I recomputed the corrected lead-3 bins through a
completely separate observation path — the deterministic zarr's own `streamflow_obs` field rather
than a fresh netCDF read — and got identical values to three decimals in all five bins. That is a
different data source and a different code path reaching the same numbers.

---

## 5. Agreement with the reviewer's corrected tables (number by number)

### 5a. Raw coverage / CRPS by lead

The reviewer's corrected table is reproduced **exactly**, on the variant that does *not* drop the
lead overhang. I ran both variants explicitly to find out where the small differences came from:

| lead | reviewer cov | mine, no drop | mine, shipped (overhang dropped) | reviewer CRPS | mine, no drop | mine, shipped |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.723 | 0.7228 ✓ | 0.7228 | 0.901 | 0.9009 ✓ | 0.9009 |
| 1 | 0.701 | 0.7015 ✓ | 0.7010 | 1.034 | 1.0337 ✓ | 1.0365 |
| 2 | 0.698 | 0.6983 ✓ | 0.6972 | 1.043 | 1.0429 ✓ | 1.0486 |
| 3 | 0.698 | 0.6979 ✓ | 0.6962 | 1.050 | 1.0503 ✓ | 1.0589 |
| 4 | 0.694 | 0.6938 ✓ | 0.6914 | 1.066 | 1.0655 ✓ | 1.0772 |
| 5 | 0.685 | 0.6853 ✓ | 0.6824 | 1.103 | 1.1028 ✓ | 1.1179 |
| 6 | 0.679 | 0.6785 ✓ | 0.6746 | 1.134 | 1.1333 (−0.001) | 1.1521 |
| 7 | 0.663 | 0.6627 ✓ | 0.6577 | 1.187 | 1.1864 (−0.001) | 1.2094 |

**All 8 coverage values agree to three decimals. 6 of 8 CRPS values agree to three decimals;
leads 6 and 7 differ by 0.001 mm/day** — last-digit rounding, not a methodological difference.

The shipped column differs from the reviewer's because I additionally drop the 308 rows per lead
(7 valid dates × 22 basins × 2 windows at lead 7) whose valid date lands in the *next training
period*. The reviewer kept them. Dropping them is the more defensible choice — those target dates
are training labels (§7) — and it moves coverage down by ≤0.005 and CRPS up by ≤0.023, i.e. the
overhang days were slightly *easier* than average, which is what early-October low flow should be.
**Neither choice changes any qualitative conclusion.** Both are documented; the CSV's
`valid_date_first/last` columns make which one is in force visible without reading the code.

### 5b. Corrected lead-3 storm bins

| storm bin | reviewer corrected bias | mine | reviewer underpred frac | mine |
|---|---:|---:|---:|---:|
| dry | +0.017 | **+0.017** ✓ | 0.507 | **0.507** ✓ |
| wet ≤P50 | +0.007 | **+0.007** ✓ | 0.501 | **0.501** ✓ |
| P50–P80 | −0.039 | **−0.039** ✓ | 0.540 | **0.540** ✓ |
| P80–P95 | −0.296 | **−0.296** ✓ | 0.734 | **0.734** ✓ |
| >P95 | −0.614 | **−0.614** ✓ | 0.879 | **0.879** ✓ |

**All 10 values agree exactly.** The reviewer's published-bias column also matches the pre-fix CSV
exactly, so their reconstruction of both the old and the new calculation is confirmed.

### 5c. Basin composition

Reviewer: 20/28 rain, 8/28 snow, cohort 14 rain + 8 snow, Mill Ck LOBO set 19 rain + 8 snow.
**Reproduced exactly** (§6).

**Verdict on task D: the reviewer's corrected numbers are reproduced.** 24 of 24 storm/basin values
exact; 14 of 16 recalibration values exact with the remaining 2 off by 0.001 from rounding. I found
no number where the reviewer was wrong.

---

## 6. Basin regime table (new: `outputs/figures/basin_regime_table.csv`)

Generator: `scripts/make_basin_regime_table.py`. One row per configured basin, with `gauge_name`,
`frac_snow`, `regime`, `area_km2`, `ele_mean_m`, `p_mean_mm_day`, `aridity_ERA5_LAND`, lat/lon,
flood-window valid-obs-day count, and the two membership flags `in_train28` / `in_cohort22`.
Regime uses the project's own rule from `make_peak_error_curve.py`: `rain` if `frac_snow < 0.3`.

**Headline: the 28 configured basins are 20 rain / 8 snow.** The 22-basin evaluation cohort is
**14 rain / 8 snow**. Holding out Mill Creek leaves a training set of **19 rain / 8 snow**.
**Mill Creek's `frac_snow` is 0.000** — it is not merely the least snowy basin, it is one of
nineteen with zero snow fraction.

Details worth carrying into the prose:

- The `frac_snow` distribution is bimodal with nothing in the middle: 19 basins at exactly 0.000,
  one at 0.177 (Salmon R at Somes Bar), then a gap to 0.466–0.794 for the eight snow basins. There
  is no marginal-classification ambiguity to argue about.
- The eight snow basins are the two Merced gauges, Bear Ck, Pitman Ck, Trinity R above Coffee Ck,
  and three Tahoe/Truckee gauges (General Ck, Blackwood Ck, Sagehen Ck).
- The rain basins are dominated by North Coast systems — Eel, Van Duzen, Mad, Smith, Trinity,
  Redwood Ck, Noyo — plus six Central Coast streams (Santa Cruz C, Lopez C, Big Sur R, Nacimiento
  R, San Lorenzo C, Arroyo Valle). Five of those six Central Coast gauges have zero flood-window
  observations and are among the six dropped from the cohort.
- `ele_mean_m` is included because the manuscript claims elevation is a static input; the flood
  config's `static_attributes` list contains area and climate indices and **no elevation**.

The review's conclusion follows: the Mill Creek LOBO *performance* failure is real and unchanged
(NSE −0.740, FHV +80%), but it cannot be explained as snow→rain regime transfer, because the
training set it was held out of is overwhelmingly rain-classified. What broke on Mill Creek is
**not established** by this experiment.

---

## 7. Split-boundary leakage, quantified (new: `outputs/figures/split_leakage.csv`)

Generator: `scripts/make_split_leakage_report.py`, which parses the periods straight out of
`configs/ca-28basin-flood-h128-config.yml`. No retraining. `lead_time: 7`, `predict_last_n: 8`, so
a sample issued at `d` carries labels for `d .. d+7`.

**28 calendar dates are both a training label and a test label:**

- Inside a nominal test window (7 days at each test-window *start*, spilled forward from the
  preceding training period): 2016-10-01…2016-10-07 and 2022-10-01…2022-10-07.
- Overhang into a training period (7 days at each test-window *end*, spilled forward from the last
  test issue dates): 2017-10-01…2017-10-07 and 2023-10-01…2023-10-07.

**14 calendar dates are both a training label and a validation label:** 2009-01-01…2009-01-07 and
2012-01-01…2012-01-07.

The review said "exactly 14 nominal test target dates — the first seven days of each test winter"
and that "lead-specific test targets overlap the next training segment" at the other side. Both
are confirmed; the full count including the overhang is 28.

**How much of the test set do those dates touch** (weighted by real observation availability, so
these are scored samples, not grid cells):

| unit | total | touched | fraction |
|---|---:|---:|---:|
| rows (basin × issue date × lead) | 127,256 | 2,464 | **1.94%** |
| samples (basin × issue date, touched if any of its 8 labels is a training label) | 15,914 | 616 | **3.87%** |
| lead-0 basin-days whose target is itself a leaked label | 15,907 | 308 | **1.94%** |
| test issue dates whose label window touches a leaked date | 730 | 28 | 3.84% |

Hand-derivation, run before the script (guardrail Rule 1): at each of the four boundaries, the
number of `(issue, lead)` slots hitting the seven leaked dates is 1+2+…+7 = 28, so 4 × 28 = 112
slots × 22 cohort basins = **2,464** rows, and 4 × 7 = 28 issue dates × 22 = **616** samples. The
script returns exactly 2,464 and 616.

**Reading.** This is small — under 2% of scored rows — and concentrated in the first week of
October, which in California is reliably low flow, so the numerical effect on any median is
almost certainly negligible. But it is not zero, it is structural rather than accidental, and
"every day used, no gaps" is the wrong objective for a 7-day horizon. The fix (a 7-day
purge/embargo on both sides of every validation and test window) requires retraining and is
deliberately not attempted here. Once done, `make_split_leakage_report.py` should be converted
into the assertion the review asked for (`test_split_target_dates_disjoint`).

One thing the row count cannot measure: the model shares weights across all eight target steps, so
a leaked label at lead 7 is not insulated from lead 0 the way an independent-per-lead model would
be. The 1.94% is a lower bound on exposure, not an upper bound on effect.

---

## 8. Claims that must change

Prose sync is deliberately **not** done. These are the edits I believe are required, with proposed
wording. Files: `paper/main.tex`, `docs/METHODS.md` §4.4/§4.5/§6, the writeup master at
the public writeup, and the site note.

### 8a. Recalibration — `paper/main.tex` ~line 305–320, METHODS §4.5

| # | Old claim | Corrected claim |
|---|---|---|
| 1 | "Coverage recovers from 0.46–0.73 to 0.84–0.93 across leads" | "Coverage recovers from a nearly lead-flat 0.66–0.72 to 0.88–0.91 across leads." |
| 2 | "the feared CRPS cost does not materialize: beyond one-day lead the correction *improves* CRPS (2.19 to 1.94 mm/day at seven days)" | "the correction is close to CRPS-neutral: the flow-conditional map costs nothing at lead 0 (0.901 → 0.900 mm/day) and gains about 1% at seven days (1.209 → 1.196), while a single global map costs 0.00–0.04 mm/day at every lead. The earlier claim of a large CRPS *gain* at long leads was an artifact of the date misalignment and is withdrawn." |
| 3 | "on the highest-flow third of days, raw coverage is 0.53 with 39% of observations escaping above the interval; the flow-conditional map reaches 0.86 with 13% above, beating a single global correction on both coverage and CRPS there" | "on the highest-flow third of days, raw coverage is 0.70 with 26% of observations escaping above the interval; the flow-conditional map reaches 0.85 with 12% above and the best CRPS of the three (2.78 vs 2.89 global, 2.85 raw), while the global map reaches higher coverage there (0.89)." |
| 4 | "high-flow coverage rises from 0.53 to 0.86 at no CRPS cost beyond one-day lead" (~line 424) | "high-flow coverage rises from 0.70 to 0.85 at essentially no CRPS cost." |
| 5 | "Calibration remains slightly under target at six-to-seven-day leads (0.84 against the 0.85–0.95 band)" | Delete. Corrected lead-7 coverage is 0.876 (flow-conditional) and 0.887 (global), both inside the 0.85–0.95 band. |
| 6 | "conformal" / "cross-window conformal fitting" wherever it appears | "cross-fitted affine calibration." There is no nonconformity score and no finite-sample coverage guarantee. |
| 7 | METHODS §4.5 fit-parameter line "mu 0.90–1.24, s 1.55–5.88 … global mu 1.08–1.14, s 1.91–2.65" | "flow-conditional mu 1.004–1.132, s 0.961–3.867; global mu 1.024–1.062, s 1.680–2.289." Note the corollary: the mu values are now all within 13% of 1, so the "all mu ≠ 1 proves displacement, not dispersion" argument is much weaker and should be softened rather than restated. |

### 8b. Storm stratification — `paper/main.tex` ~line 190–204, METHODS §4.4

| # | Old claim | Corrected claim |
|---|---|---|
| 8 | METHODS §4.4 table, lead-3 bias column: −0.038 / −0.057 / −0.010 / −0.144 / −0.532 | +0.017 / +0.007 / −0.039 / −0.296 / −0.614, and add the lead-3 `n` column (10554 / 2343 / 1534 / 931 / 413), which now differs from lead 0 because the window mask is on valid dates. |
| 9 | (absent) | Add to the §4.4 caption/notes: "Lead-3 rows pair the simulation with the observation, precipitation percentile and storm class at valid date = issue date + 3. Values published before 2026-08-19 used issue-date observations and precipitation and are withdrawn." |
| 10 | Lead-0 claims — "bias stays within ±2% of zero up to the 80th percentile", "above the 95th percentile the model under-predicts flow by 52% and runs low on 84% of days", "15,907 test basin-days" | **No change.** Lead 0 is unaffected, and the 15,907 count is independently confirmed by the closure test. |
| 11 | (implicit) the tail failure narrative | Strengthen, don't soften: correctly aligned, lead-3 underprediction above P95 is −0.614 with 88% of days low, worse than the −0.532 / 74% published. |

### 8c. Basin population and geography

| # | Old claim | Corrected claim |
|---|---|---|
| 12 | "Trained on 27 mostly-snowmelt catchments, the model learned a snowmelt mapping and applied it to a rain-driven catchment it had never seen" (~line 245) | "Held out of a 27-basin training set that is itself 19 rain-classified and 8 snow-classified (`frac_snow < 0.3`), Mill Ck's skill collapses and its high-flow bias changes sign. The failure is real; its mechanism is not identified. Regime extrapolation is one candidate among several — basin attributes, scale, flow timing, record-extension differences, spatial extrapolation, and single-seed training variation are not excluded. Establishing a regime mechanism needs leave-cluster-out tests, not one basin." |
| 13 | "Mill Ck is the only rain-driven basin in the set" | Delete. Mill Ck is one of twenty rain-classified basins and one of nineteen with `frac_snow = 0.000`. |
| 14 | Table heading "5 snowmelt focus basins" (line 117), figure caption "five snowmelt" (line 134), "high-storage snowmelt focus basins" (line 141) | "5 focus basins (4 snow-classified, 1 rain-classified)" or simply "5 focus basins" — Mill Ck is in that set and is rain-driven. |
| 15 | "snowmelt Sierra headwaters plus rain-driven coastal ranges" (map title), title "Central Valley flood-forecasting LSTM", "California Sierra case study" | The configured population is 20 rain / 8 snow, dominated by North Coast systems with six Central Coast streams, and many gauges do not drain to the Central Valley. Either rescope the prose to "a 28-gauge California daily-streamflow audit" or rebuild the cohort. This is the review's finding 2 and is a scoping decision, not a wording fix — flagging, not proposing. |
| 16 | "the fixed attribute vector includes elevation" (~line 48–51) | Remove elevation. The config's `static_attributes` are area, two aridity indices, `frac_snow`, two moisture indices, `p_mean`, and two PET means. |

### 8d. Split and status

| # | Old claim | Corrected claim |
|---|---|---|
| 17 | METHODS §3.1 "every day used, no gaps" presented as a virtue | "Periods are contiguous on issue dates. Because `lead_time = 7`, 28 calendar dates are labels in both a training and a test sample (and 14 in both training and validation), touching 1.94% of scored rows and 3.87% of scored samples. This is a known defect: a 7-day horizon needs a purged split. See `outputs/figures/split_leakage.csv`." |
| 18 | METHODS §6 "2026-08-14: the numbers are final and internally consistent." | Replace with a protocol/version status table. Two multi-lead analyses were wrong on 2026-08-14 and are corrected here; finality was premature. |
| 19 | Abstract "rerun of NOAA's operational river model" | "analysis-forced open-loop NWM retrospective." (Review finding, not touched by this memo, but it is a one-line fix and belongs on the same pass.) |

---

## 9. Ledger — what is known, what is not

**Known.**
- The zarr's `date` is the issue date; `(d, k)` predicts `d + k`. Closure-tested to one float32 ULP,
  and the wrong pairing fails by 113–136 mm/day.
- The corrected recalibration and lead-3 storm numbers in §3 and §4. Reviewer's tables reproduced.
- The 28 basins are 20 rain / 8 snow; the cohort is 14/8; the Mill Ck LOBO set is 19/8;
  Mill Ck `frac_snow` = 0.000.
- 28 shared train/test label dates; 1.94% of rows, 3.87% of samples touched.

**Not known.**
- Why Mill Creek fails. One held-out basin cannot separate regime extrapolation from attributes,
  scale, timing, record extension, spatial extrapolation, or seed variance.
- Whether the leakage matters numerically. Quantified as exposure, not as effect; measuring the
  effect requires a purged retrain.
- Whether any of this survives a second seed. Every model here is a single training run.

**Hypothesized (labelled as such).**
- The leakage effect on medians is negligible because early October is low flow. Falsifier: a
  purged retrain that moves the lead-0 median NSE by more than the ±0.01–0.03 seed band.
- The remaining raw undercoverage (0.66–0.72 against nominal 0.90) is dispersion, not displacement,
  now that the fitted mu values are close to 1. Falsifier: a pure variance-inflation map (mu fixed
  at 1) recovering coverage as well as the affine map does.

---

## 10. Files touched

**Modified (regenerated outputs):**
`scripts/make_recalibration.py`, `scripts/make_storm_stratified_skill.py`,
`outputs/figures/recalibration.csv`, `outputs/figures/recalibration.png`,
`outputs/figures/storm_stratified_skill.csv`, `outputs/figures/storm_stratified_skill.png`.

**New:**
`scripts/test_valid_date_alignment.py`, `scripts/make_basin_regime_table.py`,
`scripts/make_split_leakage_report.py`, `outputs/figures/recalibration_by_basin.csv`,
`outputs/figures/basin_regime_table.csv`, `outputs/figures/split_leakage.csv`, this memo.

**Deliberately untouched:** `paper/main.tex`, `docs/METHODS.md`, the public writeup, the site note,
everything under `runs/` and `central_valley_floodforecasting/`. No model was trained; `run evaluate`
was never invoked.

---

# Addendum: storm-stratified persistence comparison — 2026-08-19

**Question.** The writeup's "beats gauge persistence from day 2" is a whole-period, cohort-median
NSE claim. Does it survive conditioning on the storms an operator actually cares about — the
P80–P95 and >P95 precipitation bins — at leads 1–7?

**Script:** `scripts/make_storm_stratified_persistence.py` →
`outputs/figures/storm_stratified_persistence.csv`, `storm_stratified_persistence.png`.

**Method.** Valid-date aligned throughout, same convention as the corrected
`make_storm_stratified_skill.py`: a forecast issued at `d` for lead `k` is scored at `v = d + k`,
with observation, precipitation percentile, storm class and the water-year mask all taken at `v`.
Baselines are matched by lead and **imported from `make_baselines.py`, not re-implemented** — plain
persistence at lead `k` is `q(v − k)`, which is identically `q(d)`, the gauge reading at issue time;
damped persistence is `clim(v) + α^k·(q(d) − clim(d))` clipped at zero, with the 31-day smoothed
day-of-year climatology fitted on training periods and α the training lag-1 anomaly autocorrelation.
All three models are scored on an identical row set (110,117 of 111,804 rows have obs, LSTM, plain
and damped all finite), so no model is scored on days a competitor was dropped from. Leads 1–7 only:
`time_step 0` is a nowcast with no exact persistence analogue (METHODS §4.2), and inventing one was
not worth the ambiguity.

## Closure checks (all pass)

| # | check | result |
|---|---|---|
| 0 | per-basin α vs committed `persistence_by_lead.csv` | max abs diff **1.11e-16** |
| 1 | per-basin × per-lead NSE for plain and damped vs `persistence_by_lead.csv`, using that file's own masking | max abs diff **1.11e-16** for both |
| 2 | per-basin NSE win counts, aggregate | **9/22 plain and 4/22 damped at lead 1; 17/22 and 17/22 at lead 2; 21/22 and 21/22 at lead 3; 22/22 at leads 4–7** — reproduces the adversarial review's table cell for cell |
| 3 | hand check, Mill Ck lead 3 `>P95`, first 5 rows, recomputed from the raw netCDF | `valid_date = issue_date + 3` and `persistence = obs(issue_date)` confirmed on all 5; obs/LSTM/persistence match to <1e-5 |
| 4 | lead-3 row counts per bin vs corrected `storm_stratified_skill.csv` | **10554 / 2343 / 1534 / 931 / 413, delta 0 in every bin** |

Check 2 is the one that matters most: it recovers the review's per-basin win counts exactly from an
independently built table, which is why I trust the conditional numbers below.

**Dry-bin denominator caveat does not bite.** "Dry" is a *precipitation* class (< 1 mm/day), not a
flow class: mean observed flow in the dry bin is 2.63 mm/day and the smallest model MAE anywhere in
the table is 0.438 mm/day. No skill score in this table rests on a near-zero denominator, and the
script prints an explicit check to that effect. (A first run flagged seven cells; that was a bug in
my flag — it was scanning the `mae_skill_vs_*` columns, which are skill scores and legitimately
negative, alongside the raw MAE columns. Fixed; the corrected check finds nothing.)

## Results — MAE skill score `1 − MAE_LSTM/MAE_baseline` (positive = LSTM better) and win rate

| bin | lead | skill vs plain | win rate vs plain | skill vs damped | win rate vs damped |
|---|---:|---:|---:|---:|---:|
| **ALL** | 1 | −0.259 | 0.204 | −0.337 | 0.266 |
| **ALL** | 2 | +0.173 | 0.315 | +0.108 | 0.394 |
| **ALL** | 3 | +0.295 | 0.387 | +0.217 | 0.467 |
| **ALL** | 4 | +0.369 | 0.427 | +0.281 | 0.499 |
| **ALL** | 7 | +0.429 | 0.529 | +0.282 | 0.543 |
| **P80–P95** | 1 | −0.151 | 0.354 | −0.090 | 0.390 |
| **P80–P95** | 2 | **+0.056** | 0.451 | **+0.170** | 0.504 |
| **P80–P95** | 3 | +0.131 | 0.494 | +0.260 | 0.569 |
| **P80–P95** | 4 | +0.238 | 0.563 | +0.312 | 0.624 |
| **P80–P95** | 5 | +0.284 | 0.581 | +0.330 | 0.640 |
| **P80–P95** | 6 | +0.401 | 0.644 | +0.401 | 0.652 |
| **P80–P95** | 7 | +0.355 | 0.627 | +0.308 | 0.646 |
| **>P95** | 1 | −0.081 | 0.392 | −0.021 | 0.407 |
| **>P95** | 2 | **+0.064** | 0.523 | **+0.117** | 0.516 |
| **>P95** | 3 | +0.116 | 0.569 | +0.165 | 0.571 |
| **>P95** | 4 | +0.150 | 0.584 | +0.214 | 0.600 |
| **>P95** | 5 | +0.186 | 0.603 | +0.236 | 0.598 |
| **>P95** | 6 | +0.226 | 0.630 | +0.232 | 0.588 |
| **>P95** | 7 | +0.286 | 0.697 | +0.237 | 0.622 |

n per bin is lead-dependent only through the valid-date window mask: P80–P95 n = 931 and
>P95 n = 413 at every lead; ALL n = 15,863 at lead 1 falling to 15,599 at lead 7. Full table
including `dry`, `wet ≤P50`, raw MAE, relative bias and footnote pooled NSE is in the CSV.

## Plain reading

**The day-2 crossover survives in the top two storm bins on MAE, but the margin is roughly a third
of the aggregate margin and the day-count verdict lags it by two more days.** At lead 2 the LSTM's
MAE skill over plain persistence is +0.173 pooled over all days but only +0.056 in P80–P95 and
+0.064 in >P95, and its win rate against plain persistence is still *below* a coin flip in P80–P95
(0.451) — it wins on total error while losing on more days than it wins, which means the gain comes
from avoiding a few very large misses rather than from being routinely better. Against plain
persistence the win rate does not clear 0.5 in P80–P95 until lead 4 (0.563) and in >P95 until lead 2
(0.523), so the honest statement is **day 2 by MAE, day 4 by day-count in the P80–P95 bin.** By lead
7 the advantage is real and unambiguous in both bins (+0.355 and +0.286 on MAE, win rates 0.63 and
0.70). At lead 1 the gauge beats the LSTM in every bin including the extremes, as it does in the
aggregate.

**The advantage in the tail is persistence collapsing, not the LSTM performing.** In the >P95 bin
the LSTM's absolute MAE is essentially flat with lead — 10.94 mm/day at lead 1, 10.81 at lead 7,
dipping to 10.39 at lead 5 — against a bin mean flow of 17.13 mm/day, so it sits at ~63% relative
MAE at every horizon, with a relative bias of −0.58 to −0.62 that barely moves across leads. All of
the rising skill score comes from plain persistence degrading from 10.12 to 15.13 mm/day. Both
models are poor on extreme days; the LSTM is merely less poor, and its poorness is a
capacity/magnitude problem rather than a forecast-horizon one — the same peak compression §4.4 and
§4.5 describe.

## Two things worth carrying into the prose

1. **Damped persistence is the stronger null on average and the *weaker* null in storms.** Pooled
   over all days it beats plain persistence at every lead (lead 3 MAE 1.860 vs 2.065; pooled NSE
   0.458 vs 0.273). Inside P80–P95 the ordering inverts from lead 2 onward (lead 3 MAE 5.775 damped
   vs 4.913 plain; pooled NSE 0.228 vs 0.416), because damping toward climatology is exactly the
   wrong move on a storm day. So "damped persistence is the harder baseline" is a whole-period
   statement that flips on the days that matter, and the binding null in the tail is **plain**
   persistence. Any tail claim should be scored against `max(plain, damped)` per bin, not against
   damped by default.
2. **The lead-1 loss is worse than the aggregate suggests in the common bins and milder in the
   extremes.** At lead 1 the LSTM's MAE skill is −0.458 in the dry bin and −0.372 in wet ≤P50, but
   only −0.151 in P80–P95 and −0.081 in >P95. Yesterday's gauge is nearly unbeatable on a quiet day
   and merely better on a storm day — which is the argument for gauge assimilation stated in
   METHODS §4.2, now with the conditional structure attached.

**Claim wording.** Replace "the LSTM beats persistence from day 2" with: *"Pooled across all days
the LSTM overtakes plain gauge persistence on MAE at day 2 and damped persistence at day 2–3.
Conditioned on the largest storms the crossover holds at the same lead but with about a third of the
margin (MAE skill +0.06 at day 2 in both top bins), and the LSTM does not win on a majority of
individual basin-days in the P80–P95 bin until day 4. In the >P95 bin its advantage grows with lead
entirely because persistence degrades; its own error is flat at ~63% relative MAE and ~60%
under-prediction at every lead 1–7."*

**Not established here.** Whether any of this survives a second training seed (single run), whether
the ranking holds per basin rather than pooled (these are pooled basin-days, so a few large basins
can dominate a bin), and whether an ARX or gauge-assimilating model would beat both — the review's
recommendation, and the obvious next experiment.
