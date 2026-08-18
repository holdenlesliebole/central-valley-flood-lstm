"""Study-area map: the 28 training basins, regime, and cohort membership.

Spatial orientation for the writeup: where the basins sit relative to the
Sierra Nevada and the Central Valley, which are snowmelt vs rain regimes
(gauge color = Caravan frac_snow, the regime variable behind Findings 1, 3
and 5), which five are the focus basins, and which six lack flood-window
observations and are excluded from every cohort median.

California boundary: US Census cartographic boundary file, fetched once and
cached at ~/data/geo/.

Usage:  python scripts/make_map.py
Writes: outputs/figures/study_map.png
"""

from __future__ import annotations

import io
import json
import os
import urllib.request

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

ATTR = os.path.expanduser('~/data/caravan-nc/attributes/camels')
CARAVAN = os.path.expanduser('~/data/caravan-nc-extended/timeseries/netcdf/camels')
BASINS = os.path.expanduser('~/Documents/Side_projects/Hydrology/upstream/ca-basins-expanded.txt')
OUT = os.path.expanduser('~/Documents/Side_projects/Hydrology/outputs/figures')
GEO_CACHE = os.path.expanduser('~/data/geo/us_states_5m.json')

STATES_URL = ('https://eric.clst.org/assets/wiki/uploads/Stuff/'
              'gz_2010_us_040_00_5m.json')

FOCUS = {
    'camels_11264500': ('Merced @ Happy Isles', (8, -2)),
    'camels_11266500': ('Merced @ Pohono', (-96, 6)),
    'camels_11230500': ('Bear Ck', (8, -10)),
    'camels_11237500': ('Pitman Ck', (-58, -14)),
    'camels_11381500': ('Mill Ck (rain)', (10, 6)),
}

INK = '#3a3a38'


def ca_geom():
    import geopandas as gpd
    os.makedirs(os.path.dirname(GEO_CACHE), exist_ok=True)
    if not os.path.exists(GEO_CACHE):
        with urllib.request.urlopen(STATES_URL, timeout=60) as r:
            open(GEO_CACHE, 'wb').write(r.read())
    states = gpd.read_file(GEO_CACHE)
    return states[states['NAME'] == 'California']


def main() -> None:
    basins = [b.strip() for b in open(BASINS) if b.strip()]
    other = pd.read_csv(f'{ATTR}/attributes_other_camels.csv').set_index('gauge_id')
    car = pd.read_csv(f'{ATTR}/attributes_caravan_camels.csv').set_index('gauge_id')

    rows = []
    for b in basins:
        q = xr.open_dataset(f'{CARAVAN}/{b}.nc')['streamflow'].to_series()
        q.index = pd.to_datetime(q.index)
        n = (q.loc['2016-10-01':'2017-09-30'].notna().sum()
             + q.loc['2022-10-01':'2023-09-30'].notna().sum())
        rows.append({'basin': b, 'lat': other.loc[b, 'gauge_lat'],
                     'lon': other.loc[b, 'gauge_lon'],
                     'frac_snow': car.loc[b, 'frac_snow'],
                     'in_cohort': n >= 30})
    df = pd.DataFrame(rows)
    print(f'{len(df)} basins, {df.in_cohort.sum()} in cohort')

    fig, ax = plt.subplots(figsize=(6.4, 7.2))
    ca = ca_geom()
    ca.plot(ax=ax, facecolor='#f4f3ef', edgecolor='#b9b8b2', linewidth=0.9,
            zorder=0)

    cmap = plt.cm.Blues
    co = df[df.in_cohort]
    ex = df[~df.in_cohort]
    sc = ax.scatter(co.lon, co.lat, c=co.frac_snow, cmap=cmap, vmin=0, vmax=1,
                    s=46, edgecolors='#6b6b67', linewidths=0.7, zorder=3,
                    label='in the 22-basin cohort')
    ax.scatter(ex.lon, ex.lat, marker='x', color='#8a8a86', s=42,
               linewidths=1.3, zorder=3,
               label='no flood-window obs (excluded)')

    for gid, (name, (dx, dy)) in FOCUS.items():
        r = df[df.basin == gid].iloc[0]
        ax.scatter([r.lon], [r.lat], s=150, facecolors='none',
                   edgecolors=INK, linewidths=1.4, zorder=4)
        ax.annotate(name, xy=(r.lon, r.lat), xytext=(dx, dy),
                    textcoords='offset points', fontsize=8.5, color=INK,
                    fontweight='bold', zorder=5)

    ax.text(-119.7, 38.55, 'Sierra Nevada', fontsize=9, color='#8a8a86',
            style='italic', ha='center', rotation=-38)
    ax.text(-121.55, 38.05, 'Central Valley', fontsize=9, color='#8a8a86',
            style='italic', ha='center', rotation=-55)
    ax.annotate('Oroville Dam', xy=(-121.485, 39.54), fontsize=7.5,
                color='#6b6b67', xytext=(-123.55, 39.1), ha='left',
                arrowprops=dict(arrowstyle='-', color='#b9b8b2', lw=0.8))
    ax.scatter([-121.485], [39.54], marker='^', s=34, color='#6b6b67', zorder=3)

    cb = fig.colorbar(sc, ax=ax, fraction=0.033, pad=0.02)
    cb.set_label('fraction of precipitation falling as snow', fontsize=8.5)
    cb.ax.tick_params(labelsize=8)

    ax.legend(loc='upper right', frameon=False, fontsize=8, scatterpoints=1)
    ax.set_xlim(-124.6, -113.9)
    ax.set_ylim(32.4, 42.15)
    ax.set_aspect(1 / np.cos(np.deg2rad(37.5)))
    ax.set_xlabel('longitude'); ax.set_ylabel('latitude')
    ax.grid(color='#eceae6', lw=0.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.suptitle('The 28 training basins: snowmelt Sierra headwaters plus '
                 'rain-driven coastal ranges', fontsize=10.5, x=0.02, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f'{OUT}/study_map.png', dpi=150, bbox_inches='tight')
    print(f'wrote {OUT}/study_map.png')


if __name__ == '__main__':
    main()
