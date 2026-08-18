"""Event-magnitude error analysis: does peak error grow with out-of-distribution size?

Why this exists
----------------
The flood-split evaluation (METHODS.md section 4) reports pooled peak metrics
(Missed-Peaks, Peak-MAPE) over the 22-basin cohort but never asks the more
diagnostic question: does the LSTM's peak error get systematically worse for
events that exceed what the model saw in training? This script builds that
curve directly. For every observed flood event in the two held-out test
windows (WY2017, WY2023), it pairs the observed peak with the matched
simulated peak, expresses the observed peak's size relative to the basin's
training-record maximum (an out-of-distribution-ness axis), and fits a
robust (Theil-Sen) trend of relative peak error against that axis, pooled
and split by snow/rain regime (Caravan frac_snow).

Design choices, stated once here rather than re-derived at each call site:
  - Observations always come from the source Caravan netCDFs, never from
    streamflow_obs in the zarr (repo-wide rule, METHODS.md section 5.2/5.8).
  - WY2017 and WY2023 are scored as separate arrays; they are never
    concatenated, so find_peaks cannot manufacture a spurious event at the
    multi-year gap between them (same discipline as make_benchmark_flood.py).
  - The 80th-percentile prominence threshold is basin-specific but pooled
    over both test windows (one threshold per basin), while peak-finding
    itself still runs window-by-window.
  - The x-axis denominator (training-record max) uses the same train-period
    definition as the flood split (METHODS.md section 3.1): everything in
    1985-01-01..2024-10-31 outside the 2009-2011 validation window and
    outside the two test windows.

Usage:  python scripts/make_peak_error_curve.py
Writes: outputs/figures/peak_error_curve.csv
        outputs/figures/peak_error_curve.png
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import find_peaks
from scipy.stats import theilslopes

REPO = os.path.expanduser('~/Documents/Side_projects/Hydrology')
CARAVAN = os.path.expanduser('~/data/caravan-nc-extended/timeseries/netcdf/camels')
ATTR_CSV = os.path.expanduser(
    '~/data/caravan-nc/attributes/camels/attributes_caravan_camels.csv'
)
ZARR_PATH = os.path.join(
    REPO,
    'runs/ca-28basin-flood-h128_1208_025528/test/model_epoch014/test_results.zarr',
)
OUT_DIR = os.path.join(REPO, 'outputs/figures')

WINDOWS = {
    'WY2017': ('2016-10-01', '2017-09-30'),
    'WY2023': ('2022-10-01', '2023-09-30'),
}
VAL_START, VAL_END = '2009-01-01', '2011-12-31'
TRAIN_LO, TRAIN_HI = '1985-01-01', '2024-10-31'

MIN_VALID_DAYS = 30          # cohort filter: valid test-window obs days
PEAK_DISTANCE_DAYS = 5       # primary min distance between detected peaks
PEAK_DISTANCE_FALLBACK = 3   # used only if the primary setting yields <30 events
MIN_TOTAL_EVENTS = 30
MATCH_WINDOW_DAYS = 2        # sim peak = max within +/- this many days of obs peak
BINS = [0, 0.25, 0.5, 0.75, 1.0, np.inf]

INK, BLUE, ORANGE = '#3a3a38', '#2f6bd8', '#e07b2f'

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150, 'font.size': 9,
    'axes.grid': True, 'grid.color': '#e6e6e2', 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#c9c9c4', 'axes.linewidth': 0.8,
})


def obs_series(basin: str) -> pd.Series:
    """Full observed streamflow (mm/day) from the source Caravan netCDF."""
    q = xr.open_dataset(f'{CARAVAN}/{basin}.nc')['streamflow'].to_series()
    q.index = pd.to_datetime(q.index)
    return q


def train_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """True where a date is in the flood-split training period."""
    in_range = (index >= TRAIN_LO) & (index <= TRAIN_HI)
    in_val = (index >= VAL_START) & (index <= VAL_END)
    in_test = np.zeros(len(index), dtype=bool)
    for a, z in WINDOWS.values():
        in_test |= (index >= a) & (index <= z)
    return in_range & ~in_val & ~in_test


def detect_events(
    obs_win: pd.Series, prominence: float, distance: int
) -> pd.Series:
    """Peaks in an observed window series; returns obs value indexed by date."""
    clean = obs_win.dropna()
    if len(clean) < 3 or not np.isfinite(prominence):
        return pd.Series(dtype=float)
    idx, _ = find_peaks(clean.to_numpy(), prominence=prominence, distance=distance)
    return clean.iloc[idx]


def matched_sim_peak(sim_win: pd.Series, obs_date: pd.Timestamp) -> float:
    """Max simulated flow within +/- MATCH_WINDOW_DAYS of an observed peak date."""
    lo = obs_date - pd.Timedelta(days=MATCH_WINDOW_DAYS)
    hi = obs_date + pd.Timedelta(days=MATCH_WINDOW_DAYS)
    seg = sim_win.loc[lo:hi].dropna()
    return float(seg.max()) if len(seg) else np.nan


def binned_medians(df: pd.DataFrame) -> pd.DataFrame:
    cats = pd.cut(df['norm_magnitude'], BINS, right=False)
    g = df.groupby(cats, observed=True)['rel_error']
    out = g.median().to_frame('median_rel_error')
    out['n'] = g.size()
    out['bin_mid'] = [
        iv.left if np.isinf(iv.right) else (iv.left + iv.right) / 2
        for iv in out.index
    ]
    return out.reset_index(names='bin')


def fit_theil_sen(df: pd.DataFrame) -> tuple[float, float, int]:
    if len(df) < 2:
        return np.nan, np.nan, len(df)
    slope, intercept, _, _ = theilslopes(df['rel_error'], df['norm_magnitude'])
    return float(slope), float(intercept), len(df)


def main() -> None:
    res = xr.open_zarr(ZARR_PATH, consolidated=False).squeeze('freq')
    zarr_dates = pd.to_datetime(res['date'].values)
    all_basins = [str(b) for b in res['basin'].values]

    # --- cohort: >=30 valid obs days across both test windows combined ---
    obs_by_basin: dict[str, pd.Series] = {}
    cohort = []
    for gid in all_basins:
        o = obs_series(gid)
        obs_by_basin[gid] = o
        o_z = o.reindex(zarr_dates)
        n_valid = sum(o_z.loc[a:z].notna().sum() for a, z in WINDOWS.values())
        if n_valid >= MIN_VALID_DAYS:
            cohort.append(gid)
    print(f'cohort: {len(cohort)} of {len(all_basins)} basins '
          f'(>= {MIN_VALID_DAYS} valid test-window obs days)')

    # --- regime lookup (Caravan frac_snow) ---
    attr = pd.read_csv(ATTR_CSV).set_index('gauge_id')['frac_snow']
    regime = {
        gid: ('rain' if attr.loc[gid] < 0.3 else 'snow')
        for gid in cohort if gid in attr.index
    }
    missing_attr = [gid for gid in cohort if gid not in attr.index]
    if missing_attr:
        print(f'WARNING: no frac_snow attribute for {missing_attr}; dropped from regime split')

    # --- training-record max per basin (x-axis denominator) ---
    train_max = {}
    for gid in cohort:
        o = obs_by_basin[gid]
        train_max[gid] = float(o.loc[train_mask(o.index)].max())

    # --- per-basin prominence threshold: 80th pct of pooled test-window flow ---
    prominence = {}
    for gid in cohort:
        o_z = obs_by_basin[gid].reindex(zarr_dates)
        pooled = np.concatenate([
            o_z.loc[a:z].dropna().to_numpy() for a, z in WINDOWS.values()
        ])
        prominence[gid] = float(np.percentile(pooled, 80)) if len(pooled) else np.nan

    def run_pass(distance: int) -> list[dict]:
        rows = []
        for gid in cohort:
            b = res.sel(basin=gid)
            sim_full = pd.Series(
                b['streamflow_sim'].sel(time_step=0).values, index=zarr_dates
            )
            obs_full = obs_by_basin[gid].reindex(zarr_dates)
            for wname, (a, z) in WINDOWS.items():
                obs_win = obs_full.loc[a:z]
                sim_win = sim_full.loc[a:z]
                events = detect_events(obs_win, prominence[gid], distance)
                for date, obs_peak in events.items():
                    sim_peak = matched_sim_peak(sim_win, date)
                    if not np.isfinite(sim_peak) or not np.isfinite(obs_peak) \
                            or obs_peak == 0 or gid not in regime:
                        continue
                    rows.append({
                        'basin': gid,
                        'window': wname,
                        'date': date.date().isoformat(),
                        'obs_peak': obs_peak,
                        'sim_peak': sim_peak,
                        'rel_error': (sim_peak - obs_peak) / obs_peak,
                        'norm_magnitude': obs_peak / train_max[gid],
                        'regime': regime[gid],
                    })
        return rows

    distance = PEAK_DISTANCE_DAYS
    rows = run_pass(distance)
    if len(rows) < MIN_TOTAL_EVENTS:
        print(f'only {len(rows)} events at distance={distance}d; '
              f'falling back to distance={PEAK_DISTANCE_FALLBACK}d per task spec')
        distance = PEAK_DISTANCE_FALLBACK
        rows = run_pass(distance)

    df = pd.DataFrame(rows).sort_values(['basin', 'window', 'date']).reset_index(drop=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, 'peak_error_curve.csv')
    df.to_csv(csv_path, index=False)

    # --- fits ---
    slope_all, intercept_all, n_all = fit_theil_sen(df)
    fits_by_regime = {
        reg: fit_theil_sen(sub) for reg, sub in df.groupby('regime')
    }
    bins_table = binned_medians(df)

    # --- figure ---
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    colors = {'snow': BLUE, 'rain': ORANGE}
    for reg, sub in df.groupby('regime'):
        ax.scatter(sub['norm_magnitude'], sub['rel_error'], s=26,
                   color=colors[reg], alpha=0.75, edgecolor='white',
                   linewidth=0.4, label=f'{reg} basins (n={len(sub)})')
    ax.plot(bins_table['bin_mid'], bins_table['median_rel_error'],
            color=INK, marker='o', ms=6, lw=1.6, mfc=INK, mec=INK,
            label='binned median (pooled)', zorder=5)
    if np.isfinite(slope_all):
        xs = np.array([0.0, max(df['norm_magnitude'].max(), 1.0)])
        ax.plot(xs, intercept_all + slope_all * xs, color=INK, lw=1.2,
                ls='--', alpha=0.8,
                label=f'Theil-Sen (pooled): slope={slope_all:.3f}')
    ax.axhline(0, color='#8a8a86', lw=1, ls='-')
    ax.axvline(1.0, color='#c9c9c4', lw=0.9, ls=':')
    ax.text(1.01, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.05,
            'training max', fontsize=7.5, color='#6b6b67', ha='left', va='top')
    ax.set_xlabel('observed peak / basin training-record max flow')
    ax.set_ylabel('relative peak error, (sim - obs) / obs')
    ax.set_title(
        f'Peak error vs. event out-of-distribution-ness, {len(df)} events, '
        f'{df.basin.nunique()} basins, WY2017+WY2023 (min peak distance {distance}d)',
        fontsize=10)
    ax.legend(loc='best', frameon=False, fontsize=8)
    fig.tight_layout()
    png_path = os.path.join(OUT_DIR, 'peak_error_curve.png')
    fig.savefig(png_path, bbox_inches='tight')
    plt.close(fig)

    # --- report ---
    print(f'\nWrote {csv_path}')
    print(f'Wrote {png_path}')
    print(f'\nn events = {n_all}')
    print(f'pooled Theil-Sen: slope={slope_all:.4f}  intercept={intercept_all:.4f}')
    for reg in sorted(fits_by_regime):
        s, i, n = fits_by_regime[reg]
        print(f'{reg:5s} Theil-Sen: slope={s:.4f}  intercept={i:.4f}  n={n}')
    print('\nbinned medians (normalized magnitude):')
    print(bins_table.to_string(index=False))
    top3 = df.reindex(df['norm_magnitude'].sort_values(ascending=False).index).head(3)
    print('\n3 largest-magnitude events:')
    print(top3[['basin', 'window', 'date', 'obs_peak', 'sim_peak', 'rel_error',
                'norm_magnitude', 'regime']].to_string(index=False))


if __name__ == '__main__':
    main()
