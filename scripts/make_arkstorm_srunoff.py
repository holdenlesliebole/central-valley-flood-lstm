"""WRF surface-runoff reference for the ARkStorm scenario runs.

Why this exists
---------------
The scenario stress test (METHODS 4.6) found the LSTM's peaks pressed against
its training ceiling while the storm grew 1.45x. No observed truth exists for a
synthetic storm, but the scenario archive includes WRF's own Noah-MP surface
runoff for the same two storms, which provides a process-model reference for
the one quantity that does not require truth: the hist->ftr amplification. If
the physics model's runoff scales with the storm where the LSTM saturates, the
ceiling is a property of the learned mapping, not of the meteorology.

Comparability limits (state wherever quoted): SRUNOFF is instantaneous surface
runoff from the land-surface scheme -- no channel routing, no subsurface/
baseflow component -- so magnitudes and timing are not directly comparable to
gauge streamflow, and hydrograph-level comparison is not attempted. The
amplification ratio of basin-day peaks is the comparison this reference
supports.

Usage:  python scripts/make_arkstorm_srunoff.py
Reads:  ~/data/arkstorm/wrf_srunoff_hourly_3km_arkevt_30day_{hist,ftr}.nc
        outputs/figures/arkstorm_response.csv
Writes: outputs/figures/arkstorm_srunoff.csv, arkstorm_amplification.png
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_arkstorm_forcing import cohort_basins, fetch_polygons  # noqa: E402

ARK = os.path.expanduser('~/data/arkstorm')
OUT = os.path.expanduser('~/Documents/Side_projects/Hydrology/outputs/figures')

INK, BLUE, ORANGE = '#3a3a38', '#2f6bd8', '#e07b2f'


def basin_daily_srunoff(path, polys):
    from shapely.geometry import Point
    from shapely.prepared import prep
    ds = xr.open_dataset(path)
    names = [v for v in ds.data_vars if v not in ('lat', 'lon')]
    assert len(names) == 1, names
    vn = names[0]
    print(f'{os.path.basename(path)}: variable {vn}, attrs {dict(ds[vn].attrs)}')
    lat2d, lon2d = ds['lat'].values, ds['lon'].values
    daily = ds[vn].sum('hr')            # mm/hr steps -> mm/day
    vals = daily.values                 # (time, sn, we)
    out = {}
    for b, geom in polys.items():
        minx, miny, maxx, maxy = geom.bounds
        bb = ((lon2d >= minx - 0.03) & (lon2d <= maxx + 0.03)
              & (lat2d >= miny - 0.03) & (lat2d <= maxy + 0.03))
        idx = np.argwhere(bb)
        pgeom = prep(geom)
        inside = [tuple(ij) for ij in idx
                  if pgeom.contains(Point(lon2d[tuple(ij)], lat2d[tuple(ij)]))]
        if not inside:
            d2 = (lat2d - geom.centroid.y) ** 2 + (lon2d - geom.centroid.x) ** 2
            inside = [np.unravel_index(np.argmin(d2), d2.shape)]
        ij = np.array(inside)
        out[b] = vals[:, ij[:, 0], ij[:, 1]].mean(axis=1)
    return out


def main() -> None:
    basins = cohort_basins()
    polys = fetch_polygons(basins)
    sr = {}
    for name in ('hist', 'ftr'):
        sr[name] = basin_daily_srunoff(
            f'{ARK}/wrf_srunoff_hourly_3km_arkevt_30day_{name}.nc', polys)

    lstm = pd.read_csv(f'{OUT}/arkstorm_response.csv')
    lh = lstm[lstm.scenario == 'hist'].set_index('basin')
    lf = lstm[lstm.scenario == 'ftr'].set_index('basin')

    rows = []
    for b in basins:
        wh, wf = float(np.max(sr['hist'][b])), float(np.max(sr['ftr'][b]))
        rows.append({
            'basin': b,
            'wrf_srunoff_peak_hist': wh, 'wrf_srunoff_peak_ftr': wf,
            'wrf_ratio': wf / wh if wh > 0 else np.nan,
            'lstm_ratio': float(lf.loc[b, 'sim_peak'] / lh.loc[b, 'sim_peak']),
            'precip_ratio': float(lf.loc[b, 'precip_total']
                                  / lh.loc[b, 'precip_total']),
        })
    df = pd.DataFrame(rows).set_index('basin')
    df.to_csv(f'{OUT}/arkstorm_srunoff.csv')

    print('\n=== hist->ftr peak amplification: WRF surface runoff vs LSTM ===')
    print(df[['precip_ratio', 'wrf_ratio', 'lstm_ratio']].describe().loc[
        ['50%', 'mean', 'min', 'max']].to_string(
        float_format=lambda v: f'{v:.2f}'))
    print('\nper-basin:')
    print(df[['precip_ratio', 'wrf_ratio', 'lstm_ratio']].to_string(
        float_format=lambda v: f'{v:.2f}'))
    n_wrf_gt = int((df.wrf_ratio > df.precip_ratio).sum())
    print(f'\nWRF ratio > precip ratio in {n_wrf_gt}/22 basins '
          f'(runoff-fraction amplification)')

    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.scatter(df.wrf_ratio, df.lstm_ratio, s=36, color=BLUE,
               edgecolors='white', lw=0.6)
    lim = max(df.wrf_ratio.max(), df.lstm_ratio.max()) * 1.3
    ax.plot([0.8, lim], [0.8, lim], color='#c9c9c4', lw=0.9)
    med = df.precip_ratio.median()
    ax.axvline(med, color='#8a8a86', lw=0.9, ls=':')
    ax.text(med, lim * 0.8, ' median precip ratio', fontsize=7.5,
            color='#6b6b67', va='top')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.8, lim)
    ax.set_ylim(0.8, lim)
    ax.set_xlabel('WRF surface-runoff peak ratio, ARkFuture / ARkHist')
    ax.set_ylabel('LSTM streamflow peak ratio')
    ax.set_title('Scenario amplification: process-model runoff vs learned '
                 'model, 22 basins', fontsize=9.5)
    ax.grid(color='#e6e6e2', lw=0.6)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(f'{OUT}/arkstorm_amplification.png', dpi=150,
                bbox_inches='tight')
    print(f'\nwrote {OUT}/arkstorm_srunoff.csv and arkstorm_amplification.png')


if __name__ == '__main__':
    main()
