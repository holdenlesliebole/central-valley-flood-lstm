"""Quantify split-boundary target-date leakage for the 7-day flood-split config.

Why this exists
---------------
The config defines train/validation/test periods on ISSUE dates with no gaps
(`configs/ca-28basin-flood-h128-config.yml`, METHODS 3.1 "every day used, no
gaps"). But `lead_time: 7` with `predict_last_n: 8` means every sample issued at
`d` carries labels for the 8 target dates `d .. d+7`
(`datasetzoo/multimet.py::_calc_date_range(..., lead=True)`), so target dates
spill across every period boundary. The 2026-08-19 adversarial review flagged
this; this script measures exactly how much of it there is, without retraining
anything.

What it reports
---------------
1. The exact calendar dates that appear as BOTH a training target and a
   validation/test target.
2. How many test samples (basin x issue date) have at least one label date that
   is also a training label -- the "touched" fraction.
3. The same at row level (basin x issue date x lead), which is the unit every
   multi-lead table in this project scores.

Sample counts are weighted by real observation availability: a leaked label only
matters where an observation exists, so the fractions are computed over the
finite source-netCDF observations on the deterministic h128 test grid, i.e. the
same 22-basin cohort every other table uses.

This is a MEASUREMENT, not a fix. The fix is a 7-day purge/embargo on both sides
of every validation/test window, which requires retraining; once that is done
this script should be converted into the assertion the review asked for
(`test_split_target_dates_disjoint`).

Usage:  source ~/opt/anaconda3/etc/profile.d/conda.sh && conda activate googlehydrology
        cd ~/Documents/Side_projects/Hydrology
        python scripts/make_split_leakage_report.py
Writes: outputs/figures/split_leakage.csv
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import xarray as xr
import yaml

HYD = os.path.expanduser('~/Documents/Side_projects/Hydrology')
CONFIG = f'{HYD}/configs/ca-28basin-flood-h128-config.yml'
ZARR_PATH = (f'{HYD}/runs/ca-28basin-flood-h128_1208_025528/test/'
             f'model_epoch014/test_results.zarr')
CARAVAN = os.path.expanduser('~/data/caravan-nc-extended/timeseries/netcdf/camels')
OUT = f'{HYD}/outputs/figures'


def as_list(v):
    return v if isinstance(v, list) else [v]


def periods(cfg, key):
    """[(start, end), ...] issue-date periods, parsed from the DD/MM/YYYY config."""
    starts = as_list(cfg[f'{key}_start_date'])
    ends = as_list(cfg[f'{key}_end_date'])
    return [(pd.to_datetime(s, format='%d/%m/%Y'),
             pd.to_datetime(e, format='%d/%m/%Y'))
            for s, e in zip(starts, ends)]


def issue_dates(prds):
    return pd.DatetimeIndex(np.concatenate(
        [pd.date_range(a, b, freq='D').values for a, b in prds]))


def target_dates(prds, lead_time):
    """Every calendar date that appears as a label for a sample issued in prds."""
    out = set()
    for a, b in prds:
        out |= set(pd.date_range(a, b + pd.Timedelta(days=lead_time), freq='D'))
    return out


def main() -> None:
    cfg = yaml.safe_load(open(CONFIG))
    lead_time = int(cfg['lead_time'])
    n_steps = int(cfg['predict_last_n'])
    assert n_steps == lead_time + 1, (
        f'this report assumes labels are exactly d..d+lead_time '
        f'(predict_last_n={n_steps}, lead_time={lead_time})')

    tr, va, te = (periods(cfg, k) for k in ('train', 'validation', 'test'))
    print(f'lead_time={lead_time}, predict_last_n={n_steps}')
    for name, p in (('train', tr), ('validation', va), ('test', te)):
        print(f'{name:>10}: ' + ', '.join(f'{a.date()}..{b.date()}' for a, b in p))

    tr_targets = target_dates(tr, lead_time)
    va_targets = target_dates(va, lead_time)
    te_targets = target_dates(te, lead_time)

    shared_te = sorted(tr_targets & te_targets)
    shared_va = sorted(tr_targets & va_targets)
    print(f'\ntrain-target dates shared with TEST targets:       {len(shared_te)}')
    print('  ' + ', '.join(str(d.date()) for d in shared_te))
    print(f'train-target dates shared with VALIDATION targets: {len(shared_va)}')
    print('  ' + ', '.join(str(d.date()) for d in shared_va))

    te_issue = issue_dates(te)
    te_nominal = set(te_issue)     # target dates inside the nominal test windows
    inside = [d for d in shared_te if d in te_nominal]
    outside = [d for d in shared_te if d not in te_nominal]
    print(f'\n  of those, inside a nominal test window: {len(inside)} '
          f'({", ".join(str(d.date()) for d in inside)})')
    print(f'  overhang into a training period:       {len(outside)} '
          f'({", ".join(str(d.date()) for d in outside)})')

    # ---- how many scored samples/rows do the leaked labels touch? ----
    res = xr.open_zarr(ZARR_PATH, consolidated=False).squeeze('freq')
    zdates = pd.to_datetime(res['date'].values)
    basins = [str(b) for b in res['basin'].values]
    leaked = np.array(sorted(shared_te), dtype='datetime64[ns]')

    rows = []
    tot_rows = tot_rows_leaked = 0
    tot_samples = tot_samples_leaked = 0
    for bi, b in enumerate(basins):
        q = xr.open_dataset(f'{CARAVAN}/{b}.nc')['streamflow'].to_series()
        q.index = pd.to_datetime(q.index)
        # (issue date, lead) grid of finite observations at the valid date
        finite = np.zeros((len(zdates), n_steps), dtype=bool)
        is_leak = np.zeros((len(zdates), n_steps), dtype=bool)
        sim_ok = np.isfinite(res['streamflow_sim'].isel(basin=bi).values)
        for k in range(n_steps):
            vd = zdates + pd.Timedelta(days=k)
            finite[:, k] = np.isfinite(q.reindex(vd).to_numpy()) & sim_ok[:, k]
            is_leak[:, k] = np.isin(vd.values, leaked)
        n_rows = int(finite.sum())
        if n_rows == 0:
            continue
        n_rows_leak = int((finite & is_leak).sum())
        sample_has_obs = finite.any(axis=1)
        sample_leak = (finite & is_leak).any(axis=1)
        rows.append({
            'basin': b, 'n_rows': n_rows, 'n_rows_leaked': n_rows_leak,
            'frac_rows_leaked': n_rows_leak / n_rows,
            'n_samples': int(sample_has_obs.sum()),
            'n_samples_leaked': int(sample_leak.sum()),
            'frac_samples_leaked': int(sample_leak.sum()) / int(sample_has_obs.sum()),
        })
        tot_rows += n_rows; tot_rows_leaked += n_rows_leak
        tot_samples += int(sample_has_obs.sum())
        tot_samples_leaked += int(sample_leak.sum())

    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    out_csv = f'{OUT}/split_leakage.csv'
    with open(out_csv, 'w') as f:
        f.write(
            f'# Split-boundary target-date leakage, flood-split config, '
            f'lead_time={lead_time}, predict_last_n={n_steps}.\n'
            f'# A "row" is (basin, issue date, lead) with a finite observation '
            f'at valid_date = issue_date + lead; a "sample" is (basin, issue '
            f'date) and is leaked if ANY of its {n_steps} labels is also a '
            f'training label.\n'
            f'# Leaked calendar dates ({len(shared_te)} of them): '
            f'{", ".join(str(d.date()) for d in shared_te)}\n'
        )
        df.to_csv(f, index=False)
    print(f'\nWrote {out_csv}')

    leaked_set = set(shared_te)
    touched_issue = {d for d in te_issue
                     for k in range(n_steps)
                     if d + pd.Timedelta(days=k) in leaked_set}
    print(f'\ntest issue dates in the config:            {len(te_issue)}')
    print(f'issue dates whose label window touches a leaked date: '
          f'{len(touched_issue)} ({100 * len(touched_issue) / len(te_issue):.2f}%)')
    print(f'\nscored rows   (basin x issue x lead): {tot_rows:>7}   '
          f'leaked {tot_rows_leaked:>5}  ({100 * tot_rows_leaked / tot_rows:.2f}%)')
    print(f'scored samples(basin x issue):        {tot_samples:>7}   '
          f'leaked {tot_samples_leaked:>5}  '
          f'({100 * tot_samples_leaked / tot_samples:.2f}%)')

    # Lead-0-only view: the headline NSE/KGE table scores lead 0 exclusively.
    lead0_leaked = sum(1 for d in te_issue if d in leaked_set)
    print(f'\nlead-0 issue dates that are themselves leaked labels: '
          f'{lead0_leaked} of {len(te_issue)} '
          f'({100 * lead0_leaked / len(te_issue):.2f}%)')


if __name__ == '__main__':
    main()
