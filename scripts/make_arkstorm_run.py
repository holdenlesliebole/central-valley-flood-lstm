"""Run the trained flood-split LSTM on the ARkStorm 2.0 scenarios and analyze peaks.

Why this exists
---------------
Continuation of docs/arkstorm_feasibility.md and METHODS 4.4: the measured
peak-error curve ends at 0.98 of the training-record maximum. This script
runs the deterministic h128 checkpoint, unmodified, on the two scenario
forcing stores built by make_arkstorm_forcing.py (ARkHist and ARkFuture,
spliced onto the same real 2000-2002 antecedent) and reports where the
scenario response lands relative to the training record.

Framing constraint (state wherever results appear): raw WRF forcing carries
its own biases relative to the ERA5-Land/IMERG/CPC-class products the model
trained on, and forecast slots hold perfect forecasts, so outputs are a
scenario stress test of the hydrologic mapping, not a validated forecast.

Usage:  python scripts/make_arkstorm_run.py [--skip-infer]
Writes: runs/arkstorm-{hist,ftr}/ (run dirs + zarrs)
        outputs/figures/arkstorm_response.csv, arkstorm_response.png
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

REPO = os.path.expanduser('~/Documents/Side_projects/Hydrology')
RUNS = f'{REPO}/runs'
SRC_RUN = f'{RUNS}/ca-28basin-flood-h128_1208_025528'
EPOCH = 14
ARK = os.path.expanduser('~/data/arkstorm')
CARAVAN = os.path.expanduser('~/data/caravan-nc-extended/timeseries/netcdf/camels')
OUT = f'{REPO}/outputs/figures'
COHORT_FILE = f'{REPO}/configs/arkstorm-cohort-basins.txt'

SCEN_START, SCEN_END = '2002-02-09', '2002-03-12'
TEST_START, TEST_END = '09/02/2002', '31/03/2002'

INK, BLUE, ORANGE = '#3a3a38', '#2f6bd8', '#e07b2f'

TEST_WINDOWS = [('2016-10-01', '2017-09-30'), ('2022-10-01', '2023-09-30')]
VAL_WINDOW = ('2009-01-01', '2011-12-31')


def training_max(basin: str) -> float:
    """Max daily flow in the flood-split training record (as in 4.4)."""
    q = xr.open_dataset(f'{CARAVAN}/{basin}.nc')['streamflow'].to_series()
    q.index = pd.to_datetime(q.index)
    q = q.loc['1985-01-01':'2024-10-31']
    for a, b in TEST_WINDOWS + [VAL_WINDOW]:
        q = q[~((q.index >= a) & (q.index <= b))]
    return float(q.max())


def prep_run_dir(name: str) -> str:
    dst = f'{RUNS}/arkstorm-{name}'
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    shutil.copy(f'{SRC_RUN}/model_epoch{EPOCH:03d}.pt', dst)
    shutil.copy(f'{SRC_RUN}/scaler.nc', dst)
    shutil.copytree(f'{SRC_RUN}/train_data', f'{dst}/train_data')
    cfg = open(f'{SRC_RUN}/config.yml').read()
    basins = [b.strip() for b in open(COHORT_FILE)]
    assert len(basins) == 22
    import re
    cfg = re.sub(r'experiment_name: .*', f'experiment_name: arkstorm-{name}', cfg)
    cfg = re.sub(r'dynamics_data_dir: .*',
                 f'dynamics_data_dir: {ARK}/store_{name}', cfg)
    cfg = re.sub(r'test_start_date:\n(- .*\n)+',
                 f'test_start_date: {TEST_START}\n', cfg)
    cfg = re.sub(r'test_end_date:\n(- .*\n)+',
                 f'test_end_date: {TEST_END}\n', cfg)
    cfg = cfg.replace(
        'test_basin_file: \n  /Users/holden/Documents/Side_projects/Hydrology/central_valley_floodforecasting/ca-basins-expanded.txt',
        f'test_basin_file: {COHORT_FILE}')
    open(f'{dst}/config.yml', 'w').write(cfg)
    return dst


def run_infer(run_dir: str) -> None:
    env = dict(os.environ, TORCHDYNAMO_DISABLE='1',
               CLOUDSDK_CONFIG='/tmp/empty_gcloud')
    env.pop('GOOGLE_APPLICATION_CREDENTIALS', None)
    r = subprocess.run(
        ['run', 'infer', '--run-dir', run_dir, '--epoch', str(EPOCH),
         '--period', 'test'],
        cwd=REPO, env=env, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
    print(f'{run_dir}:', *tail, sep='\n  ')
    # The framework can exit nonzero from metric overflow (real 2002 obs vs
    # synthetic-storm sims are meaningless to score) while inference itself
    # completes; the zarr is the success criterion.
    zarr = f'{run_dir}/test/model_epoch{EPOCH:03d}/test_results.zarr'
    assert os.path.exists(zarr), 'infer produced no zarr'


def main() -> None:
    basins = [b.strip() for b in open(COHORT_FILE)]
    if '--skip-infer' not in sys.argv:
        for name in ('hist', 'ftr'):
            run_infer(prep_run_dir(name))

    means = {n: np.load(f'{ARK}/basin_means_{n}.npz', allow_pickle=True)
             for n in ('hist', 'ftr')}
    rows = []
    for name in ('hist', 'ftr'):
        z = f'{RUNS}/arkstorm-{name}/test/model_epoch{EPOCH:03d}/test_results.zarr'
        ds = xr.open_zarr(z, consolidated=False).squeeze('freq')
        dates = pd.to_datetime(ds['date'].values)
        sw = (dates >= SCEN_START) & (dates <= pd.Timestamp(SCEN_END) + pd.Timedelta(days=7))
        m = means[name]
        mb = list(m['basins'])
        for b in basins:
            sim = ds['streamflow_sim'].sel(basin=b).isel(time_step=0).values
            peak = float(np.nanmax(sim[sw]))
            peak_day = dates[sw][np.nanargmax(sim[sw])]
            tmax = training_max(b)
            p = m['prec'][mb.index(b)]
            rows.append({
                'scenario': name, 'basin': b, 'sim_peak': peak,
                'peak_date': peak_day, 'training_max': tmax,
                'norm_peak': peak / tmax,
                'precip_total': float(np.nansum(p)),
                'precip_max_day': float(np.nanmax(p)),
            })
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(f'{OUT}/arkstorm_response.csv', index=False)

    h = df[df.scenario == 'hist'].set_index('basin')
    f = df[df.scenario == 'ftr'].set_index('basin')
    print('\n=== ARkStorm response (deterministic h128, lead 0) ===')
    print('scenario  median norm_peak   frac basins > training max')
    for name, sub in [('hist', h), ('ftr', f)]:
        print(f'{name:>8}  {sub.norm_peak.median():15.2f}   '
              f'{(sub.norm_peak > 1).mean():25.2f}')
    pr_ratio = f.precip_total / h.precip_total
    fl_ratio = f.sim_peak / h.sim_peak
    print(f'\nftr/hist precip-total ratio: median {pr_ratio.median():.2f}')
    print(f'ftr/hist simulated-peak ratio: median {fl_ratio.median():.2f}')
    print('\nper-basin table:')
    tab = pd.DataFrame({'norm_peak_hist': h.norm_peak,
                        'norm_peak_ftr': f.norm_peak,
                        'precip_ratio': pr_ratio, 'flow_peak_ratio': fl_ratio})
    print(tab.to_string(float_format=lambda v: f'{v:.2f}'))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.6))
    a1.scatter(h.norm_peak, f.norm_peak, s=34, color=BLUE,
               edgecolors='white', lw=0.6)
    lim = max(1.05, h.norm_peak.max() * 1.1, f.norm_peak.max() * 1.1)
    a1.plot([0, lim], [0, lim], color='#c9c9c4', lw=0.9)
    a1.axvline(1, color='#8a8a86', lw=0.9, ls='--')
    a1.axhline(1, color='#8a8a86', lw=0.9, ls='--')
    a1.set_xlabel('ARkHist peak / training max')
    a1.set_ylabel('ARkFuture peak / training max')
    a1.set_title('Simulated scenario peak vs training record', fontsize=9.5)
    a2.scatter(pr_ratio, fl_ratio, s=34, color=ORANGE,
               edgecolors='white', lw=0.6)
    rlim_lo = min(0.95, pr_ratio.min() * 0.95, fl_ratio.min() * 0.95)
    rlim_hi = max(pr_ratio.max(), fl_ratio.max()) * 1.05
    a2.plot([rlim_lo, rlim_hi], [rlim_lo, rlim_hi], color='#c9c9c4', lw=0.9)
    a2.set_xlabel('precip total ratio, ARkFuture / ARkHist')
    a2.set_ylabel('simulated peak ratio')
    a2.set_title('Storm amplification vs simulated flow amplification',
                 fontsize=9.5)
    for ax in (a1, a2):
        ax.grid(color='#e6e6e2', lw=0.6)
        for s_ in ('top', 'right'):
            ax.spines[s_].set_visible(False)
    fig.suptitle('ARkStorm 2.0 scenarios through the trained model: per-basin '
                 'peaks, lead 0, 22 basins', fontsize=10.5, x=0.02, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f'{OUT}/arkstorm_response.png', dpi=150, bbox_inches='tight')
    print(f'\nwrote {OUT}/arkstorm_response.csv and arkstorm_response.png')


if __name__ == '__main__':
    main()
