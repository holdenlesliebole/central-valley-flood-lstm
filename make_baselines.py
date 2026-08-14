"""Honest-baseline suite for the CA Central Valley LSTM.

Why this exists
---------------
NSE is computed against the *mean* of the observations -- a constant. It is NOT
computed against a day-of-year climatology. Sierra snowmelt is close to the most
seasonal hydrologic signal in California, so a model can post a high NSE largely
by reproducing "water arrives in spring."

This is the same trap found independently in the morphology_ml project at Torrey
(2026-07-30): a wave-forcing convolution scored blocked-CV R2 = +0.316, but an
annual harmonic alone scored +0.363 and deseasonalized skill fell to -0.036.

So: score the binding baselines on the identical test window, basins and metric
definitions, and report the LSTM's margin over the *right* null rather than over
a constant.

Baselines
---------
mean            train-period mean flow (constant)
climatology     day-of-year mean from the training window, circularly smoothed
persistence     raw lag-1 observed flow
damped_pers.    climatology + alpha * (obs_{t-1} - clim_{t-1}), alpha fit on train

Metrics come from googlehydrology.evaluation.metrics so they are bit-for-bit the
same definitions used to score the LSTM.

Usage:  python make_baselines.py
Writes: portfolio/central_valley_figures/baseline_metrics.csv  (+ console table)
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

from googlehydrology.evaluation.metrics import kge, nse

CARAVAN = os.path.expanduser(
    '~/data/caravan-nc-extended/timeseries/netcdf/camels'
)
OUT_DIR = os.path.expanduser(
    '~/Documents/Job_Search/portfolio/central_valley_figures'
)

# Match the FLOOD-SPLIT config exactly (configs/ca-28basin-flood-h128-config.yml).
# The climatology must be fitted on TRAINING periods only, and scored on the two
# held-out flood water years. Baselines computed on the old 2012-2014 window are
# invalid alongside flood-split model results, because 2012-2014 is now training data.
TRAIN_PERIODS = [
    ('1985-01-01', '2008-12-31'),
    ('2012-01-01', '2016-09-30'),
    ('2017-10-01', '2022-09-30'),
    ('2023-10-01', '2024-10-31'),
]
TEST_PERIODS = [
    ('2016-10-01', '2017-09-30'),   # WY2017 - Oroville-spillway AR season
    ('2022-10-01', '2023-09-30'),   # WY2023 - extreme AR sequence
]
# Earliest/latest bounds, used where a continuous series is needed (persistence lags).
TEST = (TEST_PERIODS[0][0], TEST_PERIODS[-1][1])

# Circular smoothing window (days) applied to the day-of-year climatology.
# 24 years of record leaves the raw day-of-year mean noisy; 31 days is the
# conventional choice in operational hydrology. Both are reported.
SMOOTH_WINDOW = 31

FOCUS = {
    'camels_11264500': 'Merced @ Happy Isles',
    'camels_11266500': 'Merced @ Pohono',
    'camels_11230500': 'Bear Ck nr Lk Edison',
    'camels_11237500': 'Pitman Ck',
    'camels_11381500': 'Mill Ck (rain)',
}


def day_of_year_climatology(
    series: pd.Series, smooth_window: int = 0
) -> pd.Series:
    """Day-of-year mean flow, optionally smoothed circularly.

    Leap days are folded onto doy 59 (Feb 28) before averaging so that the
    climatology is defined on a fixed 1..365 index; doy 366 then reuses 365.
    """
    doy = np.asarray(series.index.dayofyear).copy()
    is_leap = np.asarray(series.index.is_leap_year)
    # On leap years, shift Mar 1 (doy 61) onward back by one so doy indexing is
    # consistent with non-leap years.
    doy[is_leap & (doy >= 61)] -= 1
    clim = pd.Series(series.to_numpy(), index=doy).groupby(level=0).mean()
    clim = clim.reindex(range(1, 366))
    clim = clim.interpolate().bfill().ffill()

    if smooth_window and smooth_window > 1:
        # Circular smoothing: wrap the year so Dec 31 and Jan 1 are neighbours.
        padded = pd.concat([clim, clim, clim])
        smoothed = padded.rolling(
            smooth_window, center=True, min_periods=1
        ).mean()
        clim = smoothed.iloc[len(clim): 2 * len(clim)]
        clim.index = range(1, 366)
    return clim


def climatology_on_index(clim: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    doy = np.asarray(index.dayofyear).copy()
    is_leap = np.asarray(index.is_leap_year)
    doy[is_leap & (doy >= 61)] -= 1
    doy = np.clip(doy, 1, 365)
    return clim.reindex(doy).to_numpy()


def select_periods(series: pd.Series, periods: list[tuple[str, str]]) -> pd.Series:
    """Concatenate the given date ranges from a series, preserving order."""
    return pd.concat([series.loc[a:b] for a, b in periods])


def score(obs: np.ndarray, sim: np.ndarray) -> tuple[float, float]:
    """NSE/KGE via the framework's own metric implementations."""
    valid = ~np.isnan(obs) & ~np.isnan(sim)
    if valid.sum() < 2:
        return np.nan, np.nan
    o = xr.DataArray(obs[valid], dims='date')
    s = xr.DataArray(sim[valid], dims='date')
    try:
        n = float(nse(o, s))
    except Exception:
        n = np.nan
    try:
        k = float(kge(o, s))
    except Exception:
        k = np.nan
    return n, k


def main() -> None:
    files = sorted(glob.glob(os.path.join(CARAVAN, '*.nc')))
    if not files:
        raise SystemExit(f'No Caravan netCDFs found under {CARAVAN}')

    rows = []
    for path in files:
        basin = os.path.basename(path).replace('.nc', '')
        ds = xr.open_dataset(path)
        q = ds['streamflow'].to_series()
        q.index = pd.to_datetime(q.index)

        train = select_periods(q, TRAIN_PERIODS)
        test = select_periods(q, TEST_PERIODS)
        if test.notna().sum() < 30 or train.notna().sum() < 365:
            rows.append({'basin': basin, 'note': 'insufficient data'})
            continue

        obs = test.to_numpy()
        idx = test.index

        clim_raw = day_of_year_climatology(train.dropna(), 0)
        clim_sm = day_of_year_climatology(train.dropna(), SMOOTH_WINDOW)
        sim_clim_raw = climatology_on_index(clim_raw, idx)
        sim_clim = climatology_on_index(clim_sm, idx)

        # Constant train-period mean.
        sim_mean = np.full_like(obs, float(train.mean()), dtype=float)

        # Raw lag-1 persistence over the *continuous* record (so the first test
        # day uses the last training-adjacent observation, not NaN).
        full = q.loc[: TEST[1]]
        sim_pers = select_periods(full.shift(1), TEST_PERIODS).to_numpy()

        # Damped persistence: alpha = lag-1 autocorrelation of training-period
        # anomalies about the smoothed climatology. Fit on TRAIN only.
        tr_clim = climatology_on_index(clim_sm, train.index)
        anom = train.to_numpy() - tr_clim
        a0, a1 = anom[:-1], anom[1:]
        m = ~np.isnan(a0) & ~np.isnan(a1)
        alpha = (
            float(np.corrcoef(a0[m], a1[m])[0, 1]) if m.sum() > 10 else np.nan
        )
        alpha = float(np.clip(alpha, 0.0, 1.0)) if np.isfinite(alpha) else 0.0

        prev_obs = select_periods(full.shift(1), TEST_PERIODS).to_numpy()
        prev_clim = climatology_on_index(
            clim_sm, idx - pd.Timedelta(days=1)
        )
        sim_damped = sim_clim + alpha * (prev_obs - prev_clim)
        sim_damped = np.clip(sim_damped, 0, None)

        row = {'basin': basin, 'name': FOCUS.get(basin, ''), 'alpha': alpha}
        for label, sim in [
            ('mean', sim_mean),
            ('clim_raw', sim_clim_raw),
            ('climatology', sim_clim),
            ('persistence', sim_pers),
            ('damped_persistence', sim_damped),
        ]:
            n, k = score(obs, sim)
            row[f'NSE_{label}'] = n
            row[f'KGE_{label}'] = k
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, 'baseline_metrics.csv')
    df.to_csv(out, index=False)

    nse_cols = [c for c in df.columns if c.startswith('NSE_')]
    print('\n=== ALL 28 BASINS: median NSE by baseline ===')
    for c in nse_cols:
        print(f'  {c[4:]:<20s} {df[c].median():+.3f}')

    focus = df[df['basin'].isin(FOCUS)].copy()
    print('\n=== 5 FOCUS BASINS: median NSE by baseline ===')
    for c in nse_cols:
        print(f'  {c[4:]:<20s} {focus[c].median():+.3f}')

    print('\n=== PER-BASIN (focus), NSE ===')
    show = ['name', 'alpha'] + nse_cols
    print(focus[show].to_string(index=False, float_format=lambda v: f'{v:+.3f}'))
    print(f'\nWrote {out}')

    lead_df = persistence_by_lead(files)
    lead_out = os.path.join(OUT_DIR, 'persistence_by_lead.csv')
    lead_df.to_csv(lead_out, index=False)
    lf = lead_df[lead_df['basin'].isin(FOCUS)]
    print('\n=== PERSISTENCE NSE vs LEAD TIME (the binding comparison) ===')
    print(f"{'lead(d)':>7}  {'all-28 median':>14}  {'focus-5 median':>15}")
    for lead in range(1, 8):
        print(
            f'{lead:>7}  {lead_df[str(lead)].median():>+14.3f}'
            f'  {lf[str(lead)].median():>+15.3f}'
        )
    print(f'Wrote {lead_out}')


def persistence_by_lead(files: list[str], max_lead: int = 7) -> pd.DataFrame:
    """Persistence NSE as a function of forecast lead time.

    This is the comparison that actually matters. The LSTM's lead-time curve is
    almost flat (all-28 median NSE 0.81 -> 0.80 over 7 days), which was read as a
    strength. Persistence decays with lead, so the honest question is not "does
    the LSTM beat persistence" but "at what lead does it overtake persistence".

    Note the asymmetry, and state it whenever these numbers are quoted: the LSTM
    is forcing-driven and never ingests observed discharge (hindcast_inputs are
    HRES/IMERG/CPC precipitation and temperature only), whereas persistence uses
    yesterday's gauge reading. Persistence therefore has information the model is
    denied -- while a real operator does have the gauge.
    """
    rows = []
    for path in files:
        basin = os.path.basename(path).replace('.nc', '')
        q = xr.open_dataset(path)['streamflow'].to_series()
        q.index = pd.to_datetime(q.index)
        full = q.loc[: TEST[1]]
        obs = select_periods(q, TEST_PERIODS).to_numpy()
        row = {'basin': basin, 'name': FOCUS.get(basin, '')}
        for lead in range(1, max_lead + 1):
            sim = select_periods(full.shift(lead), TEST_PERIODS).to_numpy()
            row[str(lead)], _ = score(obs, sim)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    main()
