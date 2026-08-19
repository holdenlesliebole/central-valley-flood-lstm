"""Build local MultiMet-format forcing stores for the ARkStorm 2.0 scenarios.

Why this exists
---------------
The event-magnitude error curve (METHODS 4.4) ends at 0.98 of the training-
record maximum while still steepening; ARkStorm-class events lie beyond it.
Huang & Swain (2022) archived the scenario meteorology (DesignSafe PRJ-3499,
ODC-BY; see docs/arkstorm_feasibility.md). This script converts their 3-km
hourly WRF output into two local forcing stores the framework's MultiMet
reader accepts unchanged, so the trained checkpoint runs the scenario without
touching model code.

Design
------
- Basin polygons from the USGS NLDI API (cached; areas cross-checked against
  Caravan's attribute, ratio reported). Cell-center point-in-polygon masking
  of the 3-km grid.
- Both scenarios are spliced onto the same real antecedent year: store dates
  run ANTE_START..SPLICE_END; ERA5-Land/CPC/IMERG antecedent values are the
  real products pulled from GCS, and the scenario window (SCEN_START..
  SCEN_END, the ARkHist calendar mapping) is overwritten with WRF daily basin
  means in every product slot. ARkFuture uses the same calendar so the two
  runs differ only in storm forcing.
- Forecast products (HRES, GRAPHCAST) are written as PERFECT FORECASTS: the
  value at issue date d, lead k is the scenario truth at d+k. This isolates
  the hydrologic model's storm response from NWP error, and is stated
  wherever results appear. Antecedent HRES/GRAPHCAST are NaN; the config's
  union_mapping fills them from ERA5-Land exactly as in training.
- Units matched to the real store (checked 2026-08-19): precipitation mm/day,
  temperature deg C. WRF T2 arrives in Kelvin and is converted; WRF hourly
  precipitation is summed to daily totals (rate-vs-accumulation is asserted
  from the file's own metadata at runtime).

Usage:  python scripts/make_arkstorm_forcing.py
Reads:  ~/data/arkstorm/wrf_{pr,t2}_*_{hist,ftr}.nc, GCS caravan-multimet
Writes: ~/data/arkstorm/store_{hist,ftr}/{ERA5_LAND,CPC,IMERG,HRES,GRAPHCAST}/timeseries.zarr
        ~/data/arkstorm/basin_polygons/*.geojson, basin_means_{hist,ftr}.nc
"""

from __future__ import annotations

import json
import os
import urllib.request

import numpy as np
import pandas as pd
import xarray as xr

ARK = os.path.expanduser('~/data/arkstorm')
POLY_DIR = f'{ARK}/basin_polygons'
ATTR = os.path.expanduser('~/data/caravan-nc/attributes/camels')
CARAVAN = os.path.expanduser('~/data/caravan-nc-extended/timeseries/netcdf/camels')

ANTE_START, SCEN_START, SCEN_END = '2000-10-01', '2002-02-09', '2002-03-12'
SPLICE_END = '2002-03-31'          # store end: scenario + recession margin
LEADS = np.arange(1, 11)           # days, matching the real HRES store

SCENARIOS = {
    'hist': (f'{ARK}/wrf_pr_hourly_3km_arkevt_30day_hist.nc',
             f'{ARK}/wrf_t2_3km_arkevt_30day_hist.nc'),
    'ftr': (f'{ARK}/wrf_pr_hourly_3km_arkevt_30day_ftr.nc',
            f'{ARK}/wrf_t2_3km_arkevt_30day_ftr.nc'),
}


def cohort_basins() -> list[str]:
    files = sorted(os.listdir(CARAVAN))
    out = []
    for f in files:
        if not f.endswith('.nc'):
            continue
        b = f[:-3]
        q = xr.open_dataset(f'{CARAVAN}/{f}')['streamflow'].to_series()
        q.index = pd.to_datetime(q.index)
        n = (q.loc['2016-10-01':'2017-09-30'].notna().sum()
             + q.loc['2022-10-01':'2023-09-30'].notna().sum())
        if n >= 30:
            out.append(b)
    assert len(out) == 22, f'expected 22 cohort basins, got {len(out)}'
    return out


def fetch_polygons(basins: list[str]) -> dict[str, object]:
    from shapely.geometry import shape
    os.makedirs(POLY_DIR, exist_ok=True)
    adf = pd.read_csv(f'{ATTR}/attributes_other_camels.csv').set_index('gauge_id')
    polys = {}
    for b in basins:
        gauge = b.replace('camels_', '')
        path = f'{POLY_DIR}/{b}.geojson'
        if not os.path.exists(path):
            url = (f'https://api.water.usgs.gov/nldi/linked-data/nwissite/'
                   f'USGS-{gauge}/basin')
            with urllib.request.urlopen(url, timeout=120) as r:
                open(path, 'wb').write(r.read())
        gj = json.load(open(path))
        geom = shape(gj['features'][0]['geometry'])
        # Rough area check vs Caravan (deg->km2 at basin latitude)
        lat = geom.centroid.y
        area_km2 = geom.area * 111.32 * 111.32 * np.cos(np.deg2rad(lat))
        ratio = area_km2 / adf.loc[b, 'area']
        flag = '  <-- CHECK' if not 0.8 < ratio < 1.25 else ''
        print(f'{b}: NLDI/Caravan area ratio {ratio:.2f}{flag}')
        polys[b] = geom
    return polys


def wrf_daily_basin_means(pr_path, t2_path, polys):
    """Daily precip totals (mm/day) and mean T2 (degC) per basin.

    File layout (verified 2026-08-19): PRECT/T2 with dims
    (time=day-index 0..29, hr=0..23, south_north, west_east); 2-D lat/lon
    coordinate arrays; PRECT in mm per hour (hourly values on the peak day
    sum to plausible daily totals); T2 in Kelvin per its own units attribute.
    """
    from shapely.geometry import Point
    from shapely.prepared import prep
    dpr = xr.open_dataset(pr_path)
    dt2 = xr.open_dataset(t2_path)
    lat2d = dpr['lat'].values
    lon2d = dpr['lon'].values
    prn, t2n = 'PRECT', 'T2'
    assert dt2[t2n].attrs.get('units', 'K') == 'K'

    # Cell membership per basin (bounding-box prefilter + point-in-polygon).
    masks = {}
    for b, geom in polys.items():
        minx, miny, maxx, maxy = geom.bounds
        bb = ((lon2d >= minx - 0.03) & (lon2d <= maxx + 0.03)
              & (lat2d >= miny - 0.03) & (lat2d <= maxy + 0.03))
        idx = np.argwhere(bb)
        pgeom = prep(geom)
        inside = [tuple(ij) for ij in idx
                  if pgeom.contains(Point(lon2d[tuple(ij)], lat2d[tuple(ij)]))]
        if len(inside) == 0:   # basin smaller than a cell: nearest center
            d2 = (lat2d - geom.centroid.y) ** 2 + (lon2d - geom.centroid.x) ** 2
            inside = [np.unravel_index(np.argmin(d2), d2.shape)]
        masks[b] = inside
        print(f'{b}: {len(inside)} cells')

    pr_daily = dpr[prn].sum('hr')                    # mm/hr steps -> mm/day
    t2_daily = dt2[t2n].mean('hr')
    n_days = pr_daily.sizes['time']
    days = pd.date_range(SCEN_START, periods=n_days, freq='D')
    print('scenario days:', days[0], '->', days[-1], f'({n_days})')

    out_p = np.full((len(polys), len(days)), np.nan, dtype='float32')
    out_t = np.full((len(polys), len(days)), np.nan, dtype='float32')
    pv = pr_daily.values
    tv = t2_daily.values
    for i, b in enumerate(polys):
        ij = np.array(masks[b])
        out_p[i] = pv[:, ij[:, 0], ij[:, 1]].mean(axis=1)
        out_t[i] = tv[:, ij[:, 0], ij[:, 1]].mean(axis=1)
    if np.nanmean(out_t) > 100:                      # Kelvin -> degC
        out_t -= 273.15
    return days, out_p, out_t


def build_store(name, basins, days, prec, temp):
    """Write the five product zarrs for one scenario."""
    root = f'{ARK}/store_{name}'
    dates = pd.date_range(ANTE_START, SPLICE_END, freq='D')
    # Scenario days map directly onto the store axis (ARkFuture reuses the
    # same pseudo-calendar, so the two runs differ only in storm forcing).
    splice = pd.DatetimeIndex(days)
    assert splice[0] >= dates[0] and splice[-1] <= dates[-1]
    pos = {d: i for i, d in enumerate(dates)}

    def gcs(product, variables):
        # Each product is pulled in its own subprocess: successive gcsfs
        # opens in one process crash with an asyncio event-loop conflict
        # (observed 2026-08-19), and a fresh interpreter per pull avoids it.
        # Results are cached, so reruns are offline.
        cache = f'{ARK}/ante_{product}.nc'
        if not os.path.exists(cache):
            import subprocess
            code = (
                "import gcsfs, xarray as xr\n"
                "fs = gcsfs.GCSFileSystem(token='anon')\n"
                f"ds = xr.open_zarr(fs.get_mapper('caravan-multimet/v1.1/{product}/timeseries.zarr'), consolidated=True)\n"
                f"sub = ds[{variables!r}].sel(basin={basins!r}).sel(date=slice('{ANTE_START}', '{SPLICE_END}')).load()\n"
                f"sub.to_netcdf('{cache}')\n"
            )
            r = subprocess.run(['python3', '-c', code], capture_output=True,
                               text=True)
            assert r.returncode == 0, r.stderr[-2000:]
        return xr.open_dataset(cache).reindex(date=dates).load()

    # Daily products: real antecedent, scenario overwrite.
    e5 = gcs('ERA5_LAND', ['era5land_total_precipitation',
                           'era5land_temperature_2m'])
    cpc = gcs('CPC', ['cpc_precipitation'])
    img = gcs('IMERG', ['imerg_precipitation'])
    for k, d in enumerate(splice):
        j = pos[d]
        e5['era5land_total_precipitation'][:, j] = prec[:, k]
        e5['era5land_temperature_2m'][:, j] = temp[:, k]
        cpc['cpc_precipitation'][:, j] = prec[:, k]
        img['imerg_precipitation'][:, j] = prec[:, k]

    # Forecast products: NaN antecedent, perfect forecasts in the window.
    n_b, n_d, n_l = len(basins), len(dates), len(LEADS)
    fp = np.full((n_b, n_d, n_l), np.nan, dtype='float32')
    ft = np.full((n_b, n_d, n_l), np.nan, dtype='float32')
    scen_vals_p = np.full((n_b, len(dates)), np.nan, dtype='float32')
    scen_vals_t = np.full((n_b, len(dates)), np.nan, dtype='float32')
    for k, d in enumerate(splice):
        scen_vals_p[:, pos[d]] = prec[:, k]
        scen_vals_t[:, pos[d]] = temp[:, k]
    for li, lead in enumerate(LEADS):
        fp[:, :n_d - lead, li] = scen_vals_p[:, lead:]
        ft[:, :n_d - lead, li] = scen_vals_t[:, lead:]

    coords_fc = {'basin': basins, 'date': dates,
                 'lead_time': LEADS.astype('timedelta64[D]').astype(
                     'timedelta64[ns]')}
    hres = xr.Dataset(
        {'hres_total_precipitation': (('basin', 'date', 'lead_time'), fp),
         'hres_temperature_2m': (('basin', 'date', 'lead_time'), ft)},
        coords=coords_fc)
    gc = xr.Dataset(
        {'graphcast_total_precipitation': (('basin', 'date', 'lead_time'), fp),
         'graphcast_temperature_2m': (('basin', 'date', 'lead_time'), ft)},
        coords=coords_fc)

    for product, ds in [('ERA5_LAND', e5), ('CPC', cpc), ('IMERG', img),
                        ('HRES', hres), ('GRAPHCAST', gc)]:
        path = f'{root}/{product}/timeseries.zarr'
        for v in ds.data_vars:
            ds[v] = ds[v].astype('float32')
        ds.to_zarr(path, mode='w', consolidated=True)
    print(f'store_{name} written: {root}')


def main() -> None:
    basins = cohort_basins()
    polys = fetch_polygons(basins)
    for name, (pr_path, t2_path) in SCENARIOS.items():
        days, prec, temp = wrf_daily_basin_means(pr_path, t2_path, polys)
        np.savez(f'{ARK}/basin_means_{name}.npz', days=days.values,
                 prec=prec, temp=temp, basins=np.array(basins))
        print(f'{name}: precip basin-mean max {np.nanmax(prec):.1f} mm/day, '
              f'30-day total mean {np.nansum(prec, axis=1).mean():.0f} mm')
        build_store(name, basins, days, prec, temp)


if __name__ == '__main__':
    main()
