"""Probabilistic scoring: CRPS, coverage, reliability, spread/skill.

Why this exists
---------------
`portfolio/central_valley_hydrology.md` names probabilistic calibration as the
project's differentiator -- "most ML-flood demos skip it, operators don't" -- and it
was the last piece still unbuilt. It is also now *necessary* rather than nice: after
the 2026-08-13 repeat-evaluation study the CMAL and deterministic models are
statistically tied on point metrics (NSE, FHV, missed peaks), so the distributional
comparison is the only thing left that can separate them.

The key idea that makes the comparison fair
-------------------------------------------
CRPS *generalizes* MAE. For a deterministic forecast -- a point mass at the predicted
value -- CRPS collapses exactly to |obs - pred|. So a deterministic model and an
ensemble model can be scored on the SAME proper scoring rule, with no special-casing.
An ensemble only beats a point forecast if its spread is genuinely informative.

Estimators
----------
Fair (unbiased-for-finite-ensemble) CRPS, the same variant WeatherNext optimises:

    CRPS = (1/m) sum_i |x_i - y|  -  1/(2 m (m-1)) sum_i sum_j |x_i - x_j|

The double sum is O(m log m) from sorted samples:

    sum_i sum_j |x_i - x_j| = 2 * sum_i (2i - m - 1) * x_(i)      (i = 1..m)

Coverage: fraction of observations inside the central `1-alpha` predictive interval.
The pre-registered target (mirroring morphology_ml) is 85-95% for a nominal 90%.

Reliability: PIT / rank histogram. For a calibrated ensemble the rank of the
observation among the sorted samples is uniform. Reported as the deviation of the
rank histogram from uniform (0 = perfect).

Spread/skill: mean ensemble standard deviation vs RMSE of the ensemble mean. A
perfectly calibrated ensemble gives ~1; <1 is under-dispersed (overconfident).

Observations come from ONE store (2026-08-14)
---------------------------------------------
The CMAL run's test_results.zarr writes corrupted `streamflow_obs` for a
nondeterministic subset of basins (recurring garbage constants like 1.966179,
values up to 1e30, phantom "valid" days on never-extended basins). This
reproduces on a clean, uncontended `run infer`, so it is NOT the MPS-contention
trap -- it is a bug in the sampling path's obs writing. The sims are unaffected
(sample medians remain hydrologically coherent against the source netCDFs).

Consequence: observations are taken ONLY from the deterministic run's zarr,
which is validated basin-by-basin against the source netCDFs at load time. Both
models are scored on the identical valid-day mask, so their n is equal by
construction.

Usage:  python make_probabilistic_scores.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import xarray as xr

RUNS = os.path.expanduser('~/Documents/Side_projects/Hydrology/runs')
OUT_DIR = os.path.expanduser(
    '~/Documents/Job_Search/portfolio/central_valley_figures'
)

CMAL = f'{RUNS}/ca-28basin-cmal-flood-h128-nantol_1208_182446/test/model_epoch016/test_results.zarr'
DETERM = f'{RUNS}/ca-28basin-flood-h128_1208_025528/test/model_epoch014/test_results.zarr'

FOCUS = {
    'camels_11264500': 'Merced @ Happy Isles',
    'camels_11266500': 'Merced @ Pohono',
    'camels_11230500': 'Bear Ck',
    'camels_11237500': 'Pitman Ck',
    'camels_11381500': 'Mill Ck (rain)',
}

NOMINAL = 0.90          # central predictive interval
MAX_SAMPLES = 2000      # thin from 7500; CRPS converges well before this
N_RANK_BINS = 20

CARAVAN = os.path.expanduser('~/data/caravan-nc-extended/timeseries/netcdf/camels')


def load_verified_obs(dt: xr.Dataset) -> dict[str, np.ndarray]:
    """Obs per basin/lead from the deterministic zarr, closure-checked vs source.

    Returns {basin: (n_lead, n_date) array}. Raises if any basin's lead-0 obs
    disagrees with the source netCDF on commonly-valid days.
    """
    dates = pd.to_datetime(dt['date'].values)
    out = {}
    for basin in [str(b) for b in dt.basin.values]:
        obs = dt['streamflow_obs'].sel(basin=basin).isel(freq=0).values  # (date, lead) or (lead, date)?
        # dims after isel: (date, time_step)
        src = xr.open_dataset(os.path.join(CARAVAN, f'{basin}.nc'))['streamflow'].to_series()
        src.index = pd.to_datetime(src.index)
        o0 = pd.Series(obs[:, 0], index=dates)
        cmp = pd.DataFrame({'src': src, 'zarr': o0}).dropna()
        # atol covers float32 quantization at near-zero flows (observed 2e-8
        # absolute error on ~6e-5 mm/day days, which fails a pure rtol test).
        if len(cmp) and not np.allclose(
            cmp['src'], cmp['zarr'], rtol=1e-4, atol=1e-6
        ):
            raise AssertionError(
                f'{basin}: deterministic-zarr obs disagrees with source netCDF '
                '-- regenerate the zarr before scoring'
            )
        out[basin] = obs.T  # (time_step, date)
    return out


def fair_crps(samples: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Fair CRPS per observation. samples (n, m) sorted internally, obs (n,)."""
    m = samples.shape[1]
    term1 = np.abs(samples - obs[:, None]).mean(axis=1)
    xs = np.sort(samples, axis=1)
    i = np.arange(1, m + 1)
    # sum_i sum_j |x_i - x_j| = 2 * sum_i (2i - m - 1) x_(i)
    pair = 2.0 * (xs * (2 * i - m - 1)).sum(axis=1)
    return term1 - pair / (2.0 * m * (m - 1))


def score_ensemble(sim: np.ndarray, obs: np.ndarray) -> dict:
    """sim (n, m) ensemble samples, obs (n,)."""
    valid = np.isfinite(obs) & np.isfinite(sim).all(axis=1)
    sim, obs = sim[valid], obs[valid]
    if len(obs) < 30:
        return {}
    crps = fair_crps(sim, obs).mean()

    lo = np.percentile(sim, 100 * (1 - NOMINAL) / 2, axis=1)
    hi = np.percentile(sim, 100 * (1 + NOMINAL) / 2, axis=1)
    coverage = float(((obs >= lo) & (obs <= hi)).mean())

    # Rank histogram / PIT: rank of obs among samples.
    ranks = (sim < obs[:, None]).sum(axis=1) / sim.shape[1]
    hist, _ = np.histogram(ranks, bins=N_RANK_BINS, range=(0, 1))
    hist = hist / hist.sum()
    reliability = float(np.abs(hist - 1.0 / N_RANK_BINS).sum() / 2)  # total variation

    ens_mean = sim.mean(axis=1)
    spread = float(sim.std(axis=1, ddof=1).mean())
    rmse = float(np.sqrt(((ens_mean - obs) ** 2).mean()))
    return {
        'CRPS': float(crps),
        'coverage90': coverage,
        'reliability_TV': reliability,
        'spread_skill': spread / rmse if rmse > 0 else np.nan,
        'n': int(len(obs)),
    }


def score_deterministic(sim: np.ndarray, obs: np.ndarray) -> dict:
    """Point forecast: CRPS reduces exactly to MAE. No interval to score."""
    valid = np.isfinite(obs) & np.isfinite(sim)
    sim, obs = sim[valid], obs[valid]
    if len(obs) < 30:
        return {}
    return {
        'CRPS': float(np.abs(sim - obs).mean()),
        'coverage90': np.nan,
        'reliability_TV': np.nan,
        'spread_skill': np.nan,
        'n': int(len(obs)),
    }


def main() -> None:
    cm = xr.open_zarr(CMAL, consolidated=False)
    dt = xr.open_zarr(DETERM, consolidated=False)
    # Obs and sims come from different stores positionally; a regenerated store
    # would misalign silently without this. The `lead` column equals the zarr
    # time_step: 0 = same-day nowcast, k = k-day-ahead forecast (fixed
    # 2026-08-17; previously labeled off by one).
    assert np.array_equal(cm['date'].values, dt['date'].values)
    assert list(cm.basin.values) == list(dt.basin.values)
    basins = [str(b) for b in cm.basin.values]
    n_lead = cm.sizes['time_step']
    step = max(1, cm.sizes['samples'] // MAX_SAMPLES)

    verified_obs = load_verified_obs(dt)

    rows = []
    for bi, basin in enumerate(basins):
        for lead in range(n_lead):
            # Obs from the verified deterministic store ONLY (see module docstring).
            obs = verified_obs[basin][lead].astype('float64')

            sim = cm['streamflow_sim'].isel(
                basin=bi, freq=0, time_step=lead
            ).values[:, ::step].astype('float64')
            dsim = dt['streamflow_sim'].isel(
                basin=bi, freq=0, time_step=lead
            ).values.astype('float64')
            if dsim.ndim > 1:
                dsim = dsim.mean(axis=-1)

            # One common mask -> both models scored on identical days.
            valid = (
                np.isfinite(obs)
                & np.isfinite(dsim)
                & np.isfinite(sim).all(axis=1)
            )
            r = score_ensemble(sim[valid], obs[valid])
            if r:
                rows.append({'model': 'CMAL', 'basin': basin,
                             'name': FOCUS.get(basin, ''), 'lead': lead, **r})
            r = score_deterministic(dsim[valid], obs[valid])
            if r:
                rows.append({'model': 'deterministic', 'basin': basin,
                             'name': FOCUS.get(basin, ''), 'lead': lead, **r})

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, 'probabilistic_scores.csv')
    df.to_csv(out, index=False)

    print('=== CRPS by lead time (median over 28 basins; LOWER IS BETTER) ===')
    print(f'{"lead":>5}{"CMAL":>12}{"determ":>12}{"CMAL better?":>15}')
    for lead in sorted(df.lead.unique()):
        c = df[(df.model == 'CMAL') & (df.lead == lead)].CRPS.median()
        d = df[(df.model == 'deterministic') & (df.lead == lead)].CRPS.median()
        print(f'{lead:>5}{c:>12.4f}{d:>12.4f}{"yes" if c < d else "no":>15}')

    cm_only = df[df.model == 'CMAL']
    print(f'\n=== CMAL calibration (nominal {NOMINAL:.0%} interval) ===')
    print(f'{"lead":>5}{"coverage":>11}{"verdict":>14}{"spread/skill":>14}{"reliab_TV":>11}')
    for lead in sorted(cm_only.lead.unique()):
        s = cm_only[cm_only.lead == lead]
        cov = s.coverage90.median()
        verdict = ('CALIBRATED' if 0.85 <= cov <= 0.95
                   else 'over-disp.' if cov > 0.95 else 'UNDER-disp.')
        print(f'{lead:>5}{cov:>11.3f}{verdict:>14}'
              f'{s.spread_skill.median():>14.3f}{s.reliability_TV.median():>11.3f}')

    print('\n=== Focus basins, lead 1 ===')
    f = df[(df.lead == 1) & (df.basin.isin(FOCUS))]
    piv = f.pivot_table(index='name', columns='model',
                        values=['CRPS', 'coverage90'])
    print(piv.to_string(float_format=lambda v: f'{v:.3f}'))
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
