"""Per-lead LSTM skill on the flood window (WY2017 + WY2023).

Why this exists
---------------
The writeup's persistence-crossover table listed the LSTM at a constant 0.819
for every lead -- a number carried over from the drought-window "flat lead-time
curve" finding, not computed per lead on the flood split. The crossover-at-day-3
claim needs the LSTM's actual per-lead skill on the same window persistence was
scored on.

Lead convention: time_step k in test_results.zarr is the forecast for day d+k
issued on day d, paired with the observation on day d+k. Persistence at lead k
uses the observation from k days before the target day. So time_step k
corresponds to persistence lead k... with one wrinkle: time_step 0 is a
same-day hindcast (lead 0), for which the natural persistence analogue is
lead 1 (yesterday's gauge). Both indexings are written out so the comparison
table can be built either way, explicitly.

Usage:  python make_lstm_by_lead.py
Writes: ~/Documents/Side_projects/Hydrology/outputs/figures/lstm_by_lead.csv
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import xarray as xr

from googlehydrology.evaluation.metrics import kge, nse

RUN_DIR = os.path.expanduser(
    '~/Documents/Side_projects/Hydrology/runs/ca-28basin-flood-h128_1208_025528'
)
EPOCH = 'model_epoch014'
OUT_DIR = os.path.expanduser(
    '~/Documents/Side_projects/Hydrology/outputs/figures'
)

FOCUS = {
    'camels_11264500': 'Merced @ Happy Isles',
    'camels_11266500': 'Merced @ Pohono',
    'camels_11230500': 'Bear Ck',
    'camels_11237500': 'Pitman Ck',
    'camels_11381500': 'Mill Ck (rain)',
}


def main() -> None:
    res = xr.open_zarr(
        os.path.join(RUN_DIR, 'test', EPOCH, 'test_results.zarr'),
        consolidated=False,
    ).squeeze('freq')

    rows = []
    for basin in [str(b) for b in res.basin.values]:
        b = res.sel(basin=basin)
        row = {'basin': basin, 'name': FOCUS.get(basin, '')}
        for k in range(res.sizes['time_step']):
            obs = b['streamflow_obs'].sel(time_step=k).values
            sim = b['streamflow_sim'].sel(time_step=k).values
            valid = np.isfinite(obs) & np.isfinite(sim)
            if valid.sum() < 30:
                row[f'NSE_ts{k}'] = np.nan
                row[f'KGE_ts{k}'] = np.nan
                continue
            o = xr.DataArray(obs[valid], dims='date')
            s = xr.DataArray(sim[valid], dims='date')
            row[f'NSE_ts{k}'] = float(nse(o, s))
            row[f'KGE_ts{k}'] = float(kge(o, s))
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, 'lstm_by_lead.csv')
    df.to_csv(out, index=False)

    focus = df[df['basin'].isin(FOCUS)]
    nse_cols = [c for c in df.columns if c.startswith('NSE_')]
    print('=== LSTM NSE by forecast time_step, flood window (WY2017+WY2023) ===')
    print(f'{"time_step":>10}{"all-28 median":>15}{"focus-5 median":>16}'
          f'{"Mill Ck":>10}')
    mill = df[df['basin'] == 'camels_11381500']
    for c in nse_cols:
        k = c.replace('NSE_ts', '')
        print(f'{k:>10}{df[c].median():>15.3f}{focus[c].median():>16.3f}'
              f'{float(mill[c].iloc[0]):>10.3f}')
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
