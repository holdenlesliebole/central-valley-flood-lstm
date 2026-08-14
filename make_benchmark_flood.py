"""Benchmark the flood-split CA LSTM against the NWM retrospective ON FLOOD YEARS.

Why this exists
---------------
The original NWM benchmark (make_benchmark.py) was scored on the 2012-2014 test
window -- the same window this project later discredited as drought-flattered.
The headline "LSTM 0.83 vs NWM 0.53" therefore rests on a period the writeup's
own Finding 0 rejects. This script closes that inconsistency by scoring both
models on the held-out flood years.

Coverage constraints (verified 2026-08-14):
  NWM v2.1 retrospective ends 2020-12-31  -> covers WY2017, not WY2023
  NWM v3.0 retrospective ends 2023-02-01  -> covers WY2017 AND WY2023 through Jan 31

So the design is:
  WY2017  (2016-10-01 .. 2017-09-30): LSTM vs NWM v2.1 vs NWM v3.0
  WY2023p (2022-10-01 .. 2023-01-31): LSTM vs NWM v3.0   [partial water year;
          includes the late-Dec 2022 - mid-Jan 2023 AR sequence, misses the
          March 2023 events -- state this caveat wherever the number appears]

Pulling v3.0 for BOTH windows removes the model-version confound; v2.1 on
WY2017 doubles as a version-sensitivity check against the old 2012-2014 result.

Windows are scored separately (never concatenated) so peak detection cannot
manufacture a spurious peak at a seam.

Fairness notes:
  - Both models are scored against the identical observed series: the
    streamflow_obs in the flood run's test_results.zarr (the extended Caravan
    record), on the common valid dates of each window.
  - The NWM retrospective is forced by observed meteorology, so it is compared
    to the LSTM's lead-0 (hindcast) simulation, not a forecast lead.
  - m3/s -> mm/day uses the Caravan basin area (same convention the LSTM
    targets are normalized by); the known Caravan-area error is <=4.5% on the
    focus basins (METHODS.md section 2.2).
  - Metrics come from googlehydrology.evaluation.metrics, bit-for-bit the same
    definitions used to score the LSTM.

Usage:  TORCHDYNAMO_DISABLE=1 python make_benchmark_flood.py
Writes: ~/Documents/Job_Search/portfolio/central_valley_figures/lstm_vs_nwm_flood_metrics.csv
        ~/data/nwm21_focus_wy2017.nc, ~/data/nwm30_focus_flood.nc (caches)
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import s3fs
import xarray as xr

from googlehydrology.evaluation.metrics import (
    fdc_fhv,
    kge,
    mean_absolute_percentage_peak_error,
    mean_peak_timing,
    missed_peaks,
    nse,
)

RUN_DIR = os.path.expanduser(
    '~/Documents/Side_projects/Hydrology/runs/ca-28basin-flood-h128_1208_025528'
)
EPOCH = 'model_epoch014'  # best-validation checkpoint (METHODS.md section 3.2)
OUT_DIR = os.path.expanduser(
    '~/Documents/Job_Search/portfolio/central_valley_figures'
)

FOCUS = {  # gauge -> (NHDPlus COMID, label)
    'camels_11264500': (21609533, 'Merced @ Happy Isles (snow)'),
    'camels_11266500': (21609641, 'Merced @ Pohono (snow)'),
    'camels_11230500': (17118323, 'Bear Ck (snow)'),
    'camels_11237500': (17116209, 'Pitman Ck (snow)'),
    'camels_11381500': (8019544, 'Mill Ck (rain)'),
}

WINDOWS = {
    'WY2017': ('2016-10-01', '2017-09-30'),
    'WY2023p': ('2022-10-01', '2023-01-31'),
}

NWM_SOURCES = {
    # name -> (zarr store, cache path, windows to pull)
    'NWM v2.1': (
        'noaa-nwm-retrospective-2-1-zarr-pds/chrtout.zarr',
        os.path.expanduser('~/data/nwm21_focus_wy2017.nc'),
        ['WY2017'],
    ),
    'NWM v3.0': (
        'noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr',
        os.path.expanduser('~/data/nwm30_focus_flood.nc'),
        ['WY2017', 'WY2023p'],
    ),
}


def pull_nwm(store: str, cache: str, windows: list[str]) -> xr.Dataset:
    """Daily-mean NWM streamflow (m3/s) for the focus COMIDs, cached locally."""
    if os.path.exists(cache):
        print(f'loaded {cache}')
        return xr.open_dataset(cache)
    # Adaptive retries: a transient DNS failure killed a 2026-08-14 pull mid-way.
    fs = s3fs.S3FileSystem(
        anon=True,
        config_kwargs={'retries': {'max_attempts': 10, 'mode': 'adaptive'}},
    )
    ds = xr.open_zarr(s3fs.S3Map(store, s3=fs), consolidated=True)
    comids = [c for c, _ in FOCUS.values()]
    parts = []
    for w in windows:
        a, b = WINDOWS[w]
        t0 = time.time()
        q = ds['streamflow'].sel(feature_id=comids).sel(time=slice(a, b))
        # Serial scheduler: keeps concurrent S3 connections (and DNS lookups) low.
        q = q.resample(time='1D').mean().compute(scheduler='synchronous')
        print(f'{store} {w}: pulled+daily in {time.time() - t0:.0f}s')
        parts.append(q)
    out = xr.concat(parts, dim='time').to_dataset(name='streamflow_m3s')
    out.to_netcdf(cache)
    return out


def score(obs: pd.Series, sim: pd.Series) -> dict[str, float]:
    """All six framework metrics on the common valid dates."""
    df = pd.DataFrame({'obs': obs, 'sim': sim}).dropna()
    o = xr.DataArray(
        df['obs'].to_numpy(), dims='date', coords={'date': df.index}
    )
    s = xr.DataArray(
        df['sim'].to_numpy(), dims='date', coords={'date': df.index}
    )
    out = {'n_days': len(df)}
    for name, fn in [
        ('NSE', nse),
        ('KGE', kge),
        ('FHV', fdc_fhv),
        ('Peak-Timing', mean_peak_timing),
        ('Missed-Peaks', missed_peaks),
        ('Peak-MAPE', mean_absolute_percentage_peak_error),
    ]:
        try:
            out[name] = float(fn(o, s))
        except Exception:
            out[name] = np.nan
    return out


def main() -> None:
    res = xr.open_zarr(
        os.path.join(RUN_DIR, 'test', EPOCH, 'test_results.zarr'),
        consolidated=False,
    ).squeeze('freq')
    dates = pd.to_datetime(res['date'].values)

    # Trap 5.2 guard: a zarr written under MPS contention carries corrupted obs.
    omin = float(np.nanmin(res['streamflow_obs'].values))
    omax = float(np.nanmax(res['streamflow_obs'].values))
    assert -1e-3 <= omin and omax < 1e4, (
        f'streamflow_obs range [{omin}, {omax}] implausible -- '
        'regenerate test_results.zarr (METHODS.md section 5.2)'
    )

    # areas (km2), Caravan convention
    adf = pd.read_csv(
        os.path.expanduser(
            '~/data/caravan-nc/attributes/camels/attributes_other_camels.csv'
        )
    )
    area = dict(zip(adf['gauge_id'], adf['area']))

    nwm = {
        name: pull_nwm(store, cache, wins)
        for name, (store, cache, wins) in NWM_SOURCES.items()
    }

    rows = []
    for gid, (comid, label) in FOCUS.items():
        b = res.sel(basin=gid)
        obs_full = pd.Series(
            b['streamflow_obs'].sel(time_step=0).values, index=dates
        )
        lstm_full = pd.Series(
            b['streamflow_sim'].sel(time_step=0).values, index=dates
        )
        for wname, (a, z) in WINDOWS.items():
            obs = obs_full.loc[a:z]
            sims = {'LSTM': lstm_full.loc[a:z]}
            for vname, ds in nwm.items():
                if wname not in NWM_SOURCES[vname][2]:
                    continue
                q = pd.Series(
                    ds['streamflow_m3s'].sel(feature_id=comid).values,
                    index=pd.to_datetime(ds['time'].values),
                ).loc[a:z]
                sims[vname] = q * 86.4 / area[gid]  # m3/s -> mm/day
            for mname, sim in sims.items():
                rows.append(
                    {'window': wname, 'basin': gid, 'name': label,
                     'model': mname, **score(obs, sim)}
                )

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, 'lstm_vs_nwm_flood_metrics.csv')
    df.to_csv(out, index=False)

    for wname in WINDOWS:
        sub = df[df['window'] == wname]
        print(f'\n=== {wname} ({WINDOWS[wname][0]} .. {WINDOWS[wname][1]}) ===')
        piv = sub.pivot_table(
            index='name', columns='model',
            values=['NSE', 'FHV', 'Peak-MAPE'], sort=False,
        )
        print(piv.to_string(float_format=lambda v: f'{v:7.3f}'))
        med = sub.groupby('model')[
            ['NSE', 'KGE', 'FHV', 'Peak-Timing', 'Missed-Peaks', 'Peak-MAPE']
        ].median()
        print('\nMEDIAN over focus basins:')
        print(med.to_string(float_format=lambda v: f'{v:7.3f}'))
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
