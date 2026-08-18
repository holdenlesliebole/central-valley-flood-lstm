"""Unified model-comparison table: all four flood-split runs, one protocol.

Protocol (the same for every row):
  - simulations read from each run's test_results.zarr (written by `run infer`,
    never `run evaluate` -- see METHODS.md on the evaluate-verb cohort trap)
  - CMAL runs summarized per day by the sample median (7500 samples; bootstrap
    std of the resulting metrics is <=0.2 in every metric, see
    make_cmal_point_metrics.py)
  - observations from the source netCDFs, never from a zarr
  - time_step 0, matching the convention of the framework's own metric logs
  - fixed cohort: basins with >=30 valid obs days in the test windows
    (22 of the 28 trained basins; 6 have no flood-window observations at all)
  - metrics from googlehydrology.evaluation.metrics

Also reports the original-split (2012-2014) run on the same 22-basin cohort so
the drought-window -> flood-window comparison is apples-to-apples.

Usage:  python make_model_comparison.py
Writes: ~/Documents/Side_projects/Hydrology/outputs/figures/model_comparison.csv
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
CARAVAN = os.path.expanduser('~/data/caravan-nc-extended/timeseries/netcdf/camels')
OUT_DIR = os.path.expanduser(
    '~/Documents/Side_projects/Hydrology/outputs/figures'
)

MODELS = {  # label -> (zarr path, is_distributional)
    'determ h16 @15': (
        f'{RUNS}/ca-28basin-flood_1208_005838/test/model_epoch015/test_results.zarr',
        False,
    ),
    'CMAL h16 @2': (
        f'{RUNS}/ca-28basin-cmal-flood_1208_014339/test/model_epoch002/test_results.zarr',
        True,
    ),
    'determ h128 @14': (
        f'{RUNS}/ca-28basin-flood-h128_1208_025528/test/model_epoch014/test_results.zarr',
        False,
    ),
    'CMAL h128 @16': (
        f'{RUNS}/ca-28basin-cmal-flood-h128-nantol_1208_182446/test/model_epoch016/test_results.zarr',
        True,
    ),
}

# Original split, drought-window test (2012-2014); no zarr survives for this
# run, but per-basin metrics are identical between the evaluate and infer code
# paths, so its test_metrics.csv is usable directly (restricted to the cohort).
ORIG_CSV = f'{RUNS}/ca-28basin_0606_120718/test/model_epoch015/test_metrics.csv'

METRICS = [
    ('NSE', nse),
    ('KGE', kge),
    ('FHV', fdc_fhv),
    ('Peak-Timing', mean_peak_timing),
    ('Missed-Peaks', missed_peaks),
    ('Peak-MAPE', mean_absolute_percentage_peak_error),
]

FOCUS = ['camels_11264500', 'camels_11266500', 'camels_11230500',
         'camels_11237500', 'camels_11381500']


WINDOWS = [('2016-10-01', '2017-09-30'), ('2022-10-01', '2023-09-30')]


def score(obs, sim, dates):
    """NSE/KGE pooled over both test windows; peak metrics per window, averaged.

    Concatenating disjoint windows would let peak detection see a spurious
    seam event (the same rule make_benchmark_flood.py enforces); NSE/KGE are
    day-wise and immune (checked: pooled vs split differ by <0.001).
    """
    valid = np.isfinite(obs) & np.isfinite(sim)
    if valid.sum() < 30:
        return {}
    o = xr.DataArray(obs[valid], dims='date', coords={'date': dates[valid]})
    s = xr.DataArray(sim[valid], dims='date', coords={'date': dates[valid]})
    out = {'n_days': int(valid.sum())}
    for name, fn in METRICS:
        if name in ('NSE', 'KGE'):
            try:
                out[name] = float(fn(o, s))
            except Exception:
                out[name] = np.nan
        else:
            vals = []
            for a, b in WINDOWS:
                m = valid & (dates >= a) & (dates <= b)
                if m.sum() < 30:
                    continue
                ow = xr.DataArray(obs[m], dims='date', coords={'date': dates[m]})
                sw = xr.DataArray(sim[m], dims='date', coords={'date': dates[m]})
                try:
                    vals.append(float(fn(ow, sw)))
                except Exception:
                    pass
            out[name] = float(np.nanmean(vals)) if vals else np.nan
    return out


def main() -> None:
    src_cache: dict[str, pd.Series] = {}

    def source_obs(basin: str) -> pd.Series:
        if basin not in src_cache:
            q = xr.open_dataset(f'{CARAVAN}/{basin}.nc')['streamflow'].to_series()
            q.index = pd.to_datetime(q.index)
            src_cache[basin] = q
        return src_cache[basin]

    # Fix the cohort from the flood-split h128 zarr's basin list + coverage.
    ref = xr.open_zarr(MODELS['determ h128 @14'][0], consolidated=False)
    ref_dates = pd.to_datetime(ref['date'].values)
    basins = [str(b) for b in ref.basin.values]
    cohort = []
    for b in basins:
        obs = source_obs(b).reindex(ref_dates)
        if obs.notna().sum() >= 30:
            cohort.append(b)
    print(f'cohort: {len(cohort)} of {len(basins)} basins have flood-window obs')

    rows = []
    for label, (path, is_dist) in MODELS.items():
        ds = xr.open_zarr(path, consolidated=False)
        dates = pd.to_datetime(ds['date'].values)
        for b in cohort:
            obs = source_obs(b).reindex(dates).to_numpy().astype('float64')
            sim = ds['streamflow_sim'].sel(basin=b).isel(
                freq=0, time_step=0
            ).values.astype('float64')
            if is_dist and sim.ndim > 1:
                sim = np.nanmedian(sim, axis=1)
            r = score(obs, sim, dates)
            if r:
                rows.append({'model': label, 'basin': b, **r})

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, 'model_comparison.csv')
    df.to_csv(out, index=False)

    cols = [n for n, _ in METRICS]
    print('\n=== Median over the fixed cohort ===')
    med = df.groupby('model')[cols].median().reindex(MODELS.keys())
    print(med.to_string(float_format=lambda v: f'{v:8.3f}'))
    print('\n=== Focus-5 median NSE ===')
    f = df[df.basin.isin(FOCUS)].groupby('model')['NSE'].median().reindex(MODELS.keys())
    print(f.to_string(float_format=lambda v: f'{v:8.3f}'))

    orig = pd.read_csv(ORIG_CSV)
    oc = orig[orig.basin.isin(cohort)]
    print('\n=== Original split (2012-2014 drought window), same cohort ===')
    print(f'  determ h16 orig-split @15: median NSE {oc.NSE.median():.3f} '
          f'(focus-5 {orig[orig.basin.isin(FOCUS)].NSE.median():.3f})')
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
