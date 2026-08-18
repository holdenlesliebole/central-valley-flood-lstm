"""Storm-conditioned skill stratification: does LSTM skill hold up in big storms?

Why this exists
----------------
Every other skill number in this project (lstm_by_lead.csv, persistence_crossover,
the NWM comparison) is a per-basin or per-lead average over the whole test window.
None of them ask the question a flood-forecasting user actually cares about: does
skill degrade as the storm gets bigger? A model that is excellent on typical days
and useless on the P95 storm day is a different product than one that is flat
across the distribution, and the aggregate NSE numbers elsewhere in this repo
cannot distinguish the two.

This script conditions each basin-day in the flood test windows (WY2017, WY2023)
on where that day's precipitation forcing sits in the basin's own climatology,
then reports bias, MAE, pooled NSE, and under-prediction frequency by storm-size
bin, at both lead 0 (same-day hindcast) and lead 3 (3-day forecast).

Precipitation source
---------------------
ERA5-Land daily total precipitation, streamed from the public Caravan-Multimet
GCS bucket (gs://caravan-multimet/v1.1/ERA5_LAND/timeseries.zarr, read
anonymously -- see the LOCAL PATCH note in
googlehydrology/datasetzoo/multimet.py:_open_zarr). ERA5_LAND is the training
config's union-fallback product (configs/ca-28basin-flood-h128-config.yml
union_mapping), i.e. the most complete of the three hindcast precip products
(HRES/IMERG/CPC), which is why it was picked over CPC for climatology: a
climatology built on a product with gaps would misrank storm days near those
gaps. The full available record (1950-01-01 .. 2024-10-31) is pulled for the
28 basins -- not just the test windows -- because "climatology percentile"
requires the full record as the reference population; the result is cached to
~/data/precip_test_windows.nc (28 basins x ~27k days, ~3MB) so reruns are
offline. The 28-basin slice sits in a single contiguous chunk of the 22485-
basin store, so the pull is a few seconds, not a bucket scan.

Percentile / bin definition
----------------------------
Per basin: days with precip < 1 mm/day are "dry" and held out of the ranking
population entirely. Every other ("wet") day in the full record is percentile-
ranked (average rank, pct=True) against the *other wet days in that basin's
full record* -- so climatology, and the P50/P80/P95 cut points, are basin-
specific and computed on the wet-day distribution only, matching the "storms
are rare among all days but ranked among storms" framing in the task spec.
Each test-window basin-day then inherits its category directly (it is itself a
member of the full-record ranking, since the record spans the test windows).

Cohort and obs
---------------
Observations are read from the per-basin Caravan-extended netCDFs, NEVER from
the run's test_results.zarr (that zarr's streamflow_obs field is not to be
trusted -- see repo CLAUDE.md hard rules), and reindexed onto the zarr's date
axis. The cohort is the 22 of 28 basins with >=30 valid obs days across the two
test windows (2016-10-01..2017-09-30, 2022-10-01..2023-09-30); the excluded 6
(camels_11124500, _11141280, _11143000, _11151300, _11176400, _11284400) have
zero obs coverage in the flood windows.

Metric notes
------------
NSE is computed with googlehydrology.evaluation.metrics.nse, pooled over all
basin-days in a bin (i.e. one NSE per bin, not an average of per-basin NSEs).
Pooling across basins with different flow magnitudes means this is NOT the
same quantity as a per-basin whole-record NSE, and a small/negative bin NSE
does not mean the model is doing anything unusual for that basin -- it means
the bin's own variance (its own mean, its own spread) sets a different bar
than the whole record's does. This is flagged again in the printed report and
carried as a text note in the CSV's header comment.

Usage:  source ~/opt/anaconda3/etc/profile.d/conda.sh && conda activate googlehydrology
        cd ~/Documents/Side_projects/Hydrology
        python scripts/make_storm_stratified_skill.py
Writes: outputs/figures/storm_stratified_skill.csv
        outputs/figures/storm_stratified_skill.png
Caches: ~/data/precip_test_windows.nc (first run streams from GCS; reruns are offline)
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from googlehydrology.evaluation.metrics import nse

RUN_DIR = os.path.expanduser(
    '~/Documents/Side_projects/Hydrology/runs/ca-28basin-flood-h128_1208_025528'
)
EPOCH = 'model_epoch014'
ZARR_PATH = os.path.join(RUN_DIR, 'test', EPOCH, 'test_results.zarr')
CARAVAN = os.path.expanduser('~/data/caravan-nc-extended/timeseries/netcdf/camels')
PRECIP_CACHE = os.path.expanduser('~/data/precip_test_windows.nc')
OUT_DIR = os.path.expanduser('~/Documents/Side_projects/Hydrology/outputs/figures')

PRECIP_BUCKET = 'gs://caravan-multimet/v1.1'
PRECIP_PRODUCT = 'ERA5_LAND'
PRECIP_VAR = 'era5land_total_precipitation'
BASIN_LIST_FILE = os.path.expanduser(
    '~/Documents/Side_projects/Hydrology/central_valley_floodforecasting/ca-basins-expanded.txt'
)

WINDOWS = [('2016-10-01', '2017-09-30'), ('2022-10-01', '2023-09-30')]
LEADS = [0, 3]
MIN_VALID_OBS_DAYS = 30
DRY_THRESHOLD_MM = 1.0

BLUE, PURPLE = '#2f6bd8', '#8458d8'
INK = '#3a3a38'

BIN_ORDER = ['dry', 'wet <=P50', 'P50-P80', 'P80-P95', '>P95']

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150, 'font.size': 9,
    'axes.grid': True, 'grid.color': '#e6e6e2', 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#c9c9c4', 'axes.linewidth': 0.8,
})


def fetch_precip_cache() -> xr.DataArray:
    """Load the cached full-record precip, streaming + caching on first run."""
    if os.path.exists(PRECIP_CACHE):
        ds = xr.open_dataset(PRECIP_CACHE)
        return ds[PRECIP_VAR]

    print(f'No cache at {PRECIP_CACHE} -- streaming from GCS ({PRECIP_BUCKET})...')
    basins = open(BASIN_LIST_FILE).read().split()
    store = f'{PRECIP_BUCKET}/{PRECIP_PRODUCT}/timeseries.zarr'
    ds = xr.open_zarr(
        store, storage_options={'token': 'anon'}, chunks='auto',
        decode_timedelta=True,
    )
    sub = ds[[PRECIP_VAR]].sel(basin=basins).load()
    os.makedirs(os.path.dirname(PRECIP_CACHE), exist_ok=True)
    sub.to_netcdf(PRECIP_CACHE)
    print(f'Cached precip for {len(basins)} basins to {PRECIP_CACHE}')
    return sub[PRECIP_VAR]


def wet_day_percentile(precip: pd.Series) -> pd.Series:
    """Percentile rank of each wet day (precip >= 1 mm/day) among wet days only.

    Returns a Series indexed like `precip`, with NaN on dry days.
    """
    wet = precip[precip >= DRY_THRESHOLD_MM]
    pct = wet.rank(method='average', pct=True)
    return pct.reindex(precip.index)


def assign_bin(is_dry: pd.Series, pct: pd.Series) -> pd.Series:
    bins = pd.Series(index=pct.index, dtype=object)
    bins[is_dry] = 'dry'
    wet = ~is_dry
    bins[wet & (pct <= 0.50)] = 'wet <=P50'
    bins[wet & (pct > 0.50) & (pct <= 0.80)] = 'P50-P80'
    bins[wet & (pct > 0.80) & (pct <= 0.95)] = 'P80-P95'
    bins[wet & (pct > 0.95)] = '>P95'
    return bins


def build_basin_day_table() -> pd.DataFrame:
    """One row per (basin, date) in the test windows, with obs, sim(lead0),
    sim(lead3), precip storm bin -- restricted to the cohort."""
    res = xr.open_zarr(ZARR_PATH, consolidated=False).squeeze('freq')
    dates = pd.to_datetime(res.date.values)
    window_mask = pd.Series(False, index=dates)
    for start, end in WINDOWS:
        window_mask |= (dates >= start) & (dates <= end)

    precip_all = fetch_precip_cache()

    rows = []
    cohort, dropped = [], []
    for basin in [str(b) for b in res.basin.values]:
        obs = xr.open_dataset(f'{CARAVAN}/{basin}.nc')['streamflow'].to_series()
        obs.index = pd.to_datetime(obs.index)
        obs = obs.reindex(dates)

        n_valid = int((obs.notna().values & window_mask.values).sum())
        if n_valid < MIN_VALID_OBS_DAYS:
            dropped.append((basin, n_valid))
            continue
        cohort.append(basin)

        precip_full = precip_all.sel(basin=basin).to_series()
        precip_full.index = pd.to_datetime(precip_full.index)
        is_dry_full = precip_full < DRY_THRESHOLD_MM
        pct_full = wet_day_percentile(precip_full)
        storm_bin_full = assign_bin(is_dry_full, pct_full)
        storm_bin = storm_bin_full.reindex(dates)

        b = res.sel(basin=basin)
        sim0 = pd.Series(b['streamflow_sim'].sel(time_step=0).values, index=dates)
        sim3 = pd.Series(b['streamflow_sim'].sel(time_step=3).values, index=dates)

        df = pd.DataFrame({
            'basin': basin, 'date': dates, 'obs': obs.values,
            'sim_ts0': sim0.values, 'sim_ts3': sim3.values,
            'storm_bin': storm_bin.values,
        })
        df = df[window_mask.values]
        rows.append(df)

    print(f'Cohort: {len(cohort)}/{len(res.basin)} basins '
          f'(>= {MIN_VALID_OBS_DAYS} valid obs days across both windows).')
    if dropped:
        names = ', '.join(f'{b} ({n})' for b, n in dropped)
        print(f'Dropped: {names}')

    return pd.concat(rows, ignore_index=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per bin x lead: n, relative bias, relative MAE, pooled NSE, under-pred freq."""
    sim_col = {0: 'sim_ts0', 3: 'sim_ts3'}
    out = []
    for storm_bin in BIN_ORDER:
        for lead in LEADS:
            sub = df[df['storm_bin'] == storm_bin][['obs', sim_col[lead]]].dropna()
            sub = sub.rename(columns={sim_col[lead]: 'sim'})
            n = len(sub)
            if n == 0:
                out.append({
                    'storm_bin': storm_bin, 'lead': lead, 'n_basin_days': 0,
                    'rel_bias': np.nan, 'rel_mae': np.nan, 'pooled_nse': np.nan,
                    'underpred_frac': np.nan,
                })
                continue
            obs_mean = sub['obs'].mean()
            rel_bias = (sub['sim'] - sub['obs']).mean() / obs_mean
            rel_mae = (sub['sim'] - sub['obs']).abs().mean() / obs_mean
            obs_da = xr.DataArray(sub['obs'].values, dims='d')
            sim_da = xr.DataArray(sub['sim'].values, dims='d')
            pooled_nse = nse(obs_da, sim_da)
            underpred_frac = (sub['sim'] < sub['obs']).mean()
            out.append({
                'storm_bin': storm_bin, 'lead': lead, 'n_basin_days': n,
                'rel_bias': rel_bias, 'rel_mae': rel_mae,
                'pooled_nse': pooled_nse, 'underpred_frac': underpred_frac,
            })
    return pd.DataFrame(out)


def make_figure(summary: pd.DataFrame) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.6))
    x = np.arange(len(BIN_ORDER))
    w = 0.36
    colors = {0: BLUE, 3: PURPLE}
    for i, lead in enumerate(LEADS):
        sub = summary[summary['lead'] == lead].set_index('storm_bin').loc[BIN_ORDER]
        offset = (i - 0.5) * w
        a1.bar(x + offset, sub['rel_bias'], w * 0.92, color=colors[lead],
               label=f'lead {lead}', edgecolor='white', linewidth=0.6)
        a2.bar(x + offset, sub['underpred_frac'], w * 0.92, color=colors[lead],
               label=f'lead {lead}', edgecolor='white', linewidth=0.6)

    a1.axhline(0, color=INK, lw=0.9)
    a1.set_xticks(x)
    a1.set_xticklabels(BIN_ORDER, rotation=18, ha='right', fontsize=8)
    a1.set_ylabel('relative bias  mean(sim-obs)/mean(obs)')
    a1.set_title('Bias by storm size', fontsize=9.5)
    a1.legend(loc='lower left', frameon=False, fontsize=8)

    a2.axhline(0.5, color='#8a8a86', lw=1, ls='--')
    a2.text(0.015, 0.965, '0.5 (unbiased)', transform=a2.transAxes, fontsize=7.5,
            color='#6b6b67', ha='left', va='top')
    a2.set_xticks(x)
    a2.set_xticklabels(BIN_ORDER, rotation=18, ha='right', fontsize=8)
    a2.set_ylim(0, 1.02)
    a2.set_ylabel('fraction of basin-days with sim < obs')
    a2.set_title('Under-prediction frequency by storm size', fontsize=9.5)
    a2.legend(loc='lower left', frameon=False, fontsize=8)

    fig.suptitle(
        'Skill by storm-size bin (per-basin wet-day precipitation percentile), '
        'WY2017 + WY2023, cohort n=22',
        fontsize=10.5, x=0.02, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = f'{OUT_DIR}/storm_stratified_skill.png'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path}')


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = build_basin_day_table()
    summary = summarize(df)

    out_csv = f'{OUT_DIR}/storm_stratified_skill.csv'
    with open(out_csv, 'w') as f:
        f.write(
            '# Storm-conditioned skill, WY2017+WY2023 flood test windows, '
            'cohort n=22 basins.\n'
            '# pooled_nse is computed over the days IN EACH BIN, pooled across '
            'basins -- it is NOT comparable to a whole-record per-basin NSE; a '
            'small bin has its own (possibly small or near-zero) obs variance, '
            'which makes NSE noisy/undefined for rare bins.\n'
            '# rel_bias = mean(sim-obs)/mean(obs); rel_mae = mean(|sim-obs|)/mean(obs); '
            'underpred_frac = fraction of basin-days with sim < obs.\n'
        )
        summary.to_csv(f, index=False)
    print(f'Wrote {out_csv}')

    make_figure(summary)

    pd.set_option('display.width', 120)
    print('\n=== Storm-stratified skill, pooled cohort (n=22 basins) ===')
    print(summary.to_string(index=False, float_format=lambda v: f'{v:.3f}'))


if __name__ == '__main__':
    main()
