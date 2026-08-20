"""Closure test: zarr observations at (issue date d, lead k) are the flow at d+k.

Why this exists
---------------
The upstream tester writes `date` as the **issue date** and `time_step` as the
lead relative to it (`googlehydrology/evaluation/tester.py`, the block that sets
`date_coords = dates[lowest_freq][:, -lead_time - 1]` and `time_step_coords +=
lead_time`), while MultiMet extracts the target sequence through `issue date +
lead_time` (`datasetzoo/multimet.py::_calc_date_range(..., lead=True)`). So

    zarr row (date = d, time_step = k)  predicts streamflow on the VALID date d + k.

Two analysis scripts silently assumed `date` was the valid date and scored every
lead against issue-date flow (`make_recalibration.py`, `make_storm_stratified_skill.py`,
both fixed 2026-08-19). This test is the regression guard: it asserts the
alignment against the source Caravan netCDFs -- never against a CMAL zarr, whose
`streamflow_obs` is corrupted (METHODS 5.2) -- and it asserts that the WRONG
alignment (issue-date flow) fails loudly for every k > 0, so the test cannot pass
vacuously on a constant or near-constant series.

Two assertions
--------------
1. ALIGNED   max |zarr_obs(d, k) - source_q(d + k)| <= float32 tolerance, all k.
2. UNALIGNED max |zarr_obs(d, k) - source_q(d)|     >> tolerance,      all k >= 1.

Tolerance is 1e-4 mm/day: zarr obs are stored float32, source netCDFs are float32
read as float64, so a value of O(100) mm/day carries ~1e-5 representation error.
The observed aligned maximum is ~8e-6 mm/day (see the printed table).

Usage:  source ~/opt/anaconda3/etc/profile.d/conda.sh && conda activate googlehydrology
        cd ~/Documents/Side_projects/Hydrology
        python scripts/test_valid_date_alignment.py
Writes: nothing. Exits non-zero on failure.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

RUN_DIR = os.path.expanduser(
    '~/Documents/Side_projects/Hydrology/runs/ca-28basin-flood-h128_1208_025528'
)
EPOCH = 'model_epoch014'
ZARR_PATH = os.path.join(RUN_DIR, 'test', EPOCH, 'test_results.zarr')
CARAVAN = os.path.expanduser('~/data/caravan-nc-extended/timeseries/netcdf/camels')

TOL = 1e-4          # mm/day; float32 storage of O(100) mm/day values
MIN_UNALIGNED = 1.0  # mm/day; the wrong pairing must be off by at least this


def source_series(basin: str) -> pd.Series:
    q = xr.open_dataset(f'{CARAVAN}/{basin}.nc')['streamflow'].to_series()
    q.index = pd.to_datetime(q.index)
    return q


def main() -> int:
    ds = xr.open_zarr(ZARR_PATH, consolidated=False).squeeze('freq')
    issue_dates = pd.to_datetime(ds['date'].values)
    basins = [str(b) for b in ds.basin.values]
    leads = [int(k) for k in ds['time_step'].values]
    print(f'zarr:   {ZARR_PATH}')
    print(f'basins: {len(basins)}   issue dates: {len(issue_dates)} '
          f'({issue_dates[0].date()} .. {issue_dates[-1].date()})   leads: {leads}')

    rows = []
    for k in leads:
        valid_dates = issue_dates + pd.Timedelta(days=k)
        err_aligned, err_unaligned, n_pairs = [], [], 0
        for bi, b in enumerate(basins):
            q = source_series(b)
            zobs = ds['streamflow_obs'].isel(basin=bi, time_step=k).values.astype('float64')
            q_valid = q.reindex(valid_dates).to_numpy().astype('float64')
            q_issue = q.reindex(issue_dates).to_numpy().astype('float64')

            m = np.isfinite(zobs) & np.isfinite(q_valid)
            if m.any():
                err_aligned.append(np.abs(zobs[m] - q_valid[m]).max())
                n_pairs += int(m.sum())
            m2 = np.isfinite(zobs) & np.isfinite(q_issue)
            if m2.any():
                err_unaligned.append(np.abs(zobs[m2] - q_issue[m2]).max())

        rows.append({
            'lead': k,
            'n_pairs': n_pairs,
            'max_abs_err_aligned': max(err_aligned) if err_aligned else np.nan,
            'max_abs_err_unaligned': max(err_unaligned) if err_unaligned else np.nan,
        })

    tab = pd.DataFrame(rows)
    print('\nmax |zarr_obs - source_q| by lead (mm/day):')
    print(tab.to_string(index=False,
                        float_format=lambda v: f'{v:.6g}'))

    ok = True
    bad = tab[tab['max_abs_err_aligned'] > TOL]
    if len(bad):
        print(f'\nFAIL: aligned closure exceeds {TOL} mm/day at leads '
              f'{list(bad["lead"])}')
        ok = False
    else:
        print(f'\nPASS: aligned closure within {TOL} mm/day at every lead '
              f'(max {tab["max_abs_err_aligned"].max():.3g}).')

    lagged = tab[tab['lead'] >= 1]
    weak = lagged[lagged['max_abs_err_unaligned'] < MIN_UNALIGNED]
    if len(weak):
        print(f'FAIL: issue-date pairing is NOT distinguishable at leads '
              f'{list(weak["lead"])} -- the test is vacuous there.')
        ok = False
    else:
        print(f'PASS: issue-date pairing is wrong by '
              f'{lagged["max_abs_err_unaligned"].min():.3g}-'
              f'{lagged["max_abs_err_unaligned"].max():.3g} mm/day at leads 1-7, '
              f'so the aligned result is not vacuous.')

    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
