"""PIT / rank-histogram diagnostic for the CMAL predictive distribution.

Why this exists
---------------
The writeup diagnosed the coverage failure (66-74% vs nominal 90%) plus
spread/skill ~ 3 as "heavy tails". Review (2026-08-17) pointed out that
conditional bias produces the same two symptoms: a distribution displaced from
the obs leaves observations outside a mis-centered interval regardless of its
width. The two stories imply different remedies (variance scaling vs bias
correction), so the PIT histogram has to decide before the diagnosis ships.

Reading the histogram (PIT = fraction of ensemble samples below the obs):
  - symmetric U-shape ......... under-dispersion (both tails escape) -> "heavy
    tails / too-narrow center" survives
  - one-sided pile-up ......... conditional bias (obs escape on one side)
  - flat ...................... calibrated

Also computed: exceedance asymmetry (obs above the 95th sample percentile vs
below the 5th) and coverage conditional on observed-flow terciles -- if the
misses concentrate in the top tercile, the failure is bias at high flows.

Obs from source netCDFs; samples from the clean CMAL zarr; ties at zero flow
are split by randomized PIT so the bottom bin is not inflated by dry days.

Usage:  python make_pit_diagnostic.py
Writes: ~/Documents/Job_Search/portfolio/central_valley_figures/pit_diagnostic.csv
        ~/Documents/Job_Search/portfolio/central_valley_figures/pit_histogram.png
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

RUNS = os.path.expanduser('~/Documents/Side_projects/Hydrology/runs')
CMAL = f'{RUNS}/ca-28basin-cmal-flood-h128-nantol_1208_182446/test/model_epoch016/test_results.zarr'
CARAVAN = os.path.expanduser('~/data/caravan-nc-extended/timeseries/netcdf/camels')
OUT = os.path.expanduser('~/Documents/Job_Search/portfolio/central_valley_figures')

N_BINS = 20
TS = 0  # nowcast, matching the point-metric convention


def main() -> None:
    cm = xr.open_zarr(CMAL, consolidated=False)
    dates = pd.to_datetime(cm['date'].values)
    rng = np.random.default_rng(42)

    pits, obs_all, above95, below05 = [], [], 0, 0
    for basin in (str(b) for b in cm.basin.values):
        src = xr.open_dataset(f'{CARAVAN}/{basin}.nc')['streamflow'].to_series()
        src.index = pd.to_datetime(src.index)
        obs = src.reindex(dates).to_numpy().astype('float64')
        sim = cm['streamflow_sim'].sel(basin=basin).isel(
            freq=0, time_step=TS
        ).values.astype('float64')
        valid = np.isfinite(obs) & np.isfinite(sim).all(axis=1)
        obs, sim = obs[valid], sim[valid]
        if len(obs) < 30:
            continue
        below = (sim < obs[:, None]).mean(axis=1)
        equal = (sim == obs[:, None]).mean(axis=1)
        pit = below + rng.uniform(0, 1, len(obs)) * equal  # randomized ties
        pits.append(pit)
        obs_all.append(obs)
        above95 += int((obs > np.percentile(sim, 95, axis=1)).sum())
        below05 += int((obs < np.percentile(sim, 5, axis=1)).sum())

    pit = np.concatenate(pits)
    obs = np.concatenate(obs_all)
    hist, edges = np.histogram(pit, bins=N_BINS, range=(0, 1), density=True)

    n = len(pit)
    print(f'n = {n} basin-days (time_step {TS})')
    print(f'obs above 95th sample pct: {above95} ({above95 / n:.1%})')
    print(f'obs below  5th sample pct: {below05} ({below05 / n:.1%})')
    print(f'PIT mean {pit.mean():.3f} (0.5 = unbiased); '
          f'frac PIT>0.9: {np.mean(pit > 0.9):.3f}, PIT<0.1: {np.mean(pit < 0.1):.3f}')

    # Coverage of the central 90% interval conditional on obs tercile.
    ter = np.quantile(obs, [1 / 3, 2 / 3])
    rows = []
    for name, mask in [
        ('low tercile', obs <= ter[0]),
        ('mid tercile', (obs > ter[0]) & (obs <= ter[1])),
        ('high tercile', obs > ter[1]),
    ]:
        inside = (pit >= 0.05) & (pit <= 0.95)
        cov = float(inside[mask].mean())
        hi_esc = float((pit[mask] > 0.95).mean())
        lo_esc = float((pit[mask] < 0.05).mean())
        rows.append({'flow band': name, 'coverage90': cov,
                     'escape above': hi_esc, 'escape below': lo_esc})
        print(f'{name:>13s}: coverage {cov:.3f} | escapes above {hi_esc:.3f} '
              f'below {lo_esc:.3f}')
    pd.DataFrame(rows).to_csv(f'{OUT}/pit_diagnostic.csv', index=False)

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.bar(edges[:-1], hist, width=np.diff(edges), align='edge',
           color='#8458d8', edgecolor='white', linewidth=0.6)
    ax.axhline(1.0, color='#8a8a86', lw=1, ls='--')
    ax.text(0.99, 1.03, 'uniform = calibrated', fontsize=7.5, color='#6b6b67',
            ha='right', transform=ax.get_yaxis_transform())
    ax.set_xlabel('PIT (rank of observation in the ensemble)')
    ax.set_ylabel('density')
    ax.set_title('CMAL rank histogram, all basins, nowcast', fontsize=9.5)
    ax.grid(axis='y', color='#e6e6e2', lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(f'{OUT}/pit_histogram.png', dpi=150, bbox_inches='tight')
    print(f'wrote {OUT}/pit_histogram.png and pit_diagnostic.csv')


if __name__ == '__main__':
    main()
