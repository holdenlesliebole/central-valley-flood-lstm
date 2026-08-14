"""CMAL point metrics from the stored sample ensemble, on verified observations.

Why this exists
---------------
The framework's own CMAL evaluation path writes corrupted `streamflow_obs` for a
nondeterministic subset of basins (see make_probabilistic_scores.py, 2026-08-14),
and the same in-memory buffer feeds its metric computation -- so `run evaluate`
CMAL metrics cannot be trusted even on an uncontended GPU. The 2026-08-13
"10 repeated evaluations" table inherited this: its KGE (0.690 +/- 0.029) sits
2.7 sigma from a clean evaluation's 0.770.

This script replaces that protocol. It reads the CMAL sample ensemble from the
(clean) zarr once, summarizes each day by the sample median -- the framework's
own point summary for distributional heads -- and scores it against observations
taken from the verified deterministic store. Sampling spread is quantified by
bootstrap resampling over the sample axis (the same source of variance that made
repeated `run evaluate` calls disagree), which needs no GPU and cannot be
contaminated.

Metrics at time_step 0, matching the convention of every test_metrics.csv
number in METHODS.md.

Usage:  python make_cmal_point_metrics.py
Writes: ~/Documents/Job_Search/portfolio/central_valley_figures/cmal_point_metrics.csv
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import xarray as xr

from googlehydrology.evaluation.metrics import (
    fdc_fhv,
    kge,
    mean_absolute_percentage_peak_error,
    mean_peak_timing,
    missed_peaks,
    nse,
)

RUNS = os.path.expanduser('~/Documents/Side_projects/Hydrology/runs')
CMAL = f'{RUNS}/ca-28basin-cmal-flood-h128-nantol_1208_182446/test/model_epoch016/test_results.zarr'
DETERM = f'{RUNS}/ca-28basin-flood-h128_1208_025528/test/model_epoch014/test_results.zarr'
CARAVAN = os.path.expanduser('~/data/caravan-nc-extended/timeseries/netcdf/camels')
OUT_DIR = os.path.expanduser(
    '~/Documents/Job_Search/portfolio/central_valley_figures'
)

N_BOOT = 10
RNG_SEED = 42

METRICS = [
    ('NSE', nse),
    ('KGE', kge),
    ('FHV', fdc_fhv),
    ('Peak-Timing', mean_peak_timing),
    ('Missed-Peaks', missed_peaks),
    ('Peak-MAPE', mean_absolute_percentage_peak_error),
]


def score(obs: np.ndarray, sim: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    valid = np.isfinite(obs) & np.isfinite(sim)
    if valid.sum() < 30:
        return {}
    o = xr.DataArray(obs[valid], dims='date', coords={'date': dates[valid]})
    s = xr.DataArray(sim[valid], dims='date', coords={'date': dates[valid]})
    out = {}
    for name, fn in METRICS:
        try:
            out[name] = float(fn(o, s))
        except Exception:
            out[name] = np.nan
    return out


def main() -> None:
    cm = xr.open_zarr(CMAL, consolidated=False)
    dates = pd.to_datetime(cm['date'].values)
    rng = np.random.default_rng(RNG_SEED)

    # Test-window mask: obs outside WY2017+WY2023 exist in the source record but
    # were never forecast; restrict scoring to the forecast dates.
    in_test = (
        ((dates >= '2016-10-01') & (dates <= '2017-09-30'))
        | ((dates >= '2022-10-01') & (dates <= '2023-09-30'))
    )

    rows = []
    for bi, basin in enumerate(str(b) for b in cm.basin.values):
        # Obs directly from the source netCDFs -- never from a sampling run's
        # zarr (see module docstring).
        src = xr.open_dataset(f'{CARAVAN}/{basin}.nc')['streamflow'].to_series()
        src.index = pd.to_datetime(src.index)
        obs = src.reindex(dates).to_numpy().astype('float64')
        obs[~in_test] = np.nan

        samples = cm['streamflow_sim'].sel(basin=basin).isel(
            freq=0, time_step=0
        ).values.astype('float64')  # (date, samples)
        n_samp = samples.shape[1]

        med = np.nanmedian(samples, axis=1)
        r = score(obs, med, dates)
        if not r:
            continue
        row = {'basin': basin, **r}

        # Bootstrap over the sample axis: spread of the point summary under
        # re-drawn ensembles, mirroring repeat-evaluation variance.
        boot = {name: [] for name, _ in METRICS}
        for _ in range(N_BOOT):
            idx = rng.integers(0, n_samp, n_samp)
            bmed = np.nanmedian(samples[:, idx], axis=1)
            br = score(obs, bmed, dates)
            for name, _ in METRICS:
                boot[name].append(br.get(name, np.nan))
        for name, _ in METRICS:
            row[f'{name}_bootstd'] = float(np.nanstd(boot[name]))
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, 'cmal_point_metrics.csv')
    df.to_csv(out, index=False)

    dt_metrics = pd.read_csv(
        os.path.join(os.path.dirname(DETERM), '..', 'test_metrics.csv')
        if False else
        f'{RUNS}/ca-28basin-flood-h128_1208_025528/test/model_epoch014/test_metrics.csv'
    )
    print('=== CMAL (sample-median point summary, verified obs, time_step 0) ===')
    print('median over 28 basins, +/- median bootstrap std:')
    for name, _ in METRICS:
        print(f'  {name:<14s} {df[name].median():+8.3f}  '
              f'(boot std {df[f"{name}_bootstd"].median():.3f})')
    print('\n=== Deterministic reference (test_metrics.csv, same convention) ===')
    for name in ['NSE', 'KGE', 'FHV', 'Peak-Timing', 'Missed-Peaks', 'Peak-MAPE']:
        if name in dt_metrics:
            print(f'  {name:<14s} {dt_metrics[name].median():+8.3f}')
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
