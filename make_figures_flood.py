"""Figures for the five-findings flood-split writeup (2026-08-14).

One figure per finding, from the final CSVs/zarrs (see METHODS.md section 4).
Colors are fixed per entity across all figures: observed = ink, LSTM = blue,
NWM = orange (v2.1 = lighter/dashed), CMAL = purple, persistence = orange.
Palette CVD-validated (dataviz validator, all checks pass).

Usage:  python make_figures_flood.py
Writes: ~/Documents/Job_Search/portfolio/central_valley_figures/*.png
"""

from __future__ import annotations

import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

RUNS = os.path.expanduser('~/Documents/Side_projects/Hydrology/runs')
OUT = os.path.expanduser('~/Documents/Job_Search/portfolio/central_valley_figures')
CARAVAN = os.path.expanduser('~/data/caravan-nc-extended/timeseries/netcdf/camels')

INK, BLUE, ORANGE, ORANGE2, PURPLE = '#3a3a38', '#2f6bd8', '#e07b2f', '#f0a86e', '#8458d8'

FOCUS = {
    'camels_11264500': 'Merced @ Happy Isles',
    'camels_11266500': 'Merced @ Pohono',
    'camels_11230500': 'Bear Ck',
    'camels_11237500': 'Pitman Ck',
    'camels_11381500': 'Mill Ck (rain)',
}

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150, 'font.size': 9,
    'axes.grid': True, 'grid.color': '#e6e6e2', 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#c9c9c4', 'axes.linewidth': 0.8,
})


def obs_series(basin: str) -> pd.Series:
    q = xr.open_dataset(f'{CARAVAN}/{basin}.nc')['streamflow'].to_series()
    q.index = pd.to_datetime(q.index)
    return q


def sim_ts0(zarr_path: str, basin: str) -> pd.Series:
    ds = xr.open_zarr(zarr_path, consolidated=False)
    v = ds['streamflow_sim'].sel(basin=basin).isel(freq=0, time_step=0).values
    if v.ndim > 1:
        v = np.nanmedian(v, axis=1)
    return pd.Series(v, index=pd.to_datetime(ds['date'].values))


DET = f'{RUNS}/ca-28basin-flood-h128_1208_025528/test/model_epoch014/test_results.zarr'


def fig_hydrographs() -> None:
    """Finding 2: WY2017 hydrographs, obs vs LSTM — peaks under-predicted."""
    fig, axes = plt.subplots(5, 1, figsize=(8.2, 9), sharex=True)
    for ax, (gid, name) in zip(axes, FOCUS.items()):
        o = obs_series(gid).loc['2016-10-01':'2017-09-30']
        s = sim_ts0(DET, gid).loc['2016-10-01':'2017-09-30']
        ax.plot(o.index, o, color=INK, lw=1.1, label='observed')
        ax.plot(s.index, s, color=BLUE, lw=1.4, alpha=0.9, label='LSTM (lead 0)')
        pk = o.idxmax()
        ax.margins(y=0.14)
        # Annotate away from the basin-name label: text to the right when the
        # peak sits in the left third of the window.
        left = (pk - o.index[0]) / (o.index[-1] - o.index[0]) < 0.33
        ax.annotate(f'obs peak {o.max():.0f}', xy=(pk, o.max()),
                    xytext=(6 if left else -6, 0), textcoords='offset points',
                    ha='left' if left else 'right',
                    fontsize=7.5, color='#6b6b67')
        ax.set_ylabel('mm/day')
        ax.text(0.005, 0.86, name, transform=ax.transAxes, fontsize=9,
                fontweight='bold', color=INK)
        ax.margins(x=0.004)
    axes[0].legend(loc='center right', frameon=False, fontsize=8)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.suptitle('Held-out WY2017 (Oroville AR season): timing is right, magnitude is not',
                 fontsize=10.5, x=0.02, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(f'{OUT}/flood_hydrographs.png', bbox_inches='tight')
    plt.close(fig)


def fig_persistence() -> None:
    """Finding 1: skill vs lead — persistence decays, the LSTM holds."""
    lstm = pd.read_csv(f'{OUT}/lstm_by_lead.csv')
    pers = pd.read_csv(f'{OUT}/persistence_by_lead.csv')
    focus = list(FOCUS)
    leads = range(1, 8)
    panels = [
        ('All basins (n=22)', lstm, pers, None),
        ('5 snowmelt focus basins', lstm[lstm.basin.isin(focus)],
         pers[pers.basin.isin(focus)], None),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True)
    for ax, (title, lf, pf, _) in zip(axes, panels):
        lv = [lf[f'NSE_ts{k}'].median() for k in leads]
        pv = [pf[str(k)].median() for k in leads]
        dv = [pf[f'damped_{k}'].median() for k in leads]
        ax.plot(list(leads), lv, color=BLUE, lw=2, marker='o', ms=4, label='LSTM')
        ax.plot(list(leads), pv, color=ORANGE, lw=2, marker='o', ms=4,
                label='gauge persistence')
        ax.plot(list(leads), dv, color=ORANGE2, lw=1.8, ls='--', marker='o',
                ms=3.5, label='damped persistence')
        ax.text(leads[-1], lv[-1], '  LSTM', color=BLUE, va='center', fontsize=8.5)
        ax.text(leads[-1], pv[-1], '  persistence', color='#b35d1a', va='center',
                fontsize=8.5)
        cross = next((k for k, (a, b) in enumerate(zip(lv, pv), 1) if a > b), None)
        cross_d = next((k for k, (a, b) in enumerate(zip(lv, dv), 1) if a > b), None)
        if cross:
            ax.axvline(cross, color='#c9c9c4', lw=0.9, ls=':')
            label = (f'crossover day {cross}' if cross_d == cross
                     else f'crossover day {cross} (plain)')
            ax.text(cross + 0.15, 0.97, label,
                    transform=ax.get_xaxis_transform(), fontsize=7.5,
                    color='#6b6b67', ha='left', va='top')
        if cross_d and cross_d != cross:
            ax.axvline(cross_d, color='#c9c9c4', lw=0.9, ls=':')
            ax.text(cross_d + 0.15, 0.90, f'day {cross_d} (damped)',
                    transform=ax.get_xaxis_transform(), fontsize=7.5,
                    color='#6b6b67', ha='left', va='top')
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel('forecast lead (days)')
        ax.set_xlim(0.7, 7.9)
    axes[0].set_ylabel('median NSE')
    axes[0].legend(loc='lower left', frameon=False, fontsize=8)
    fig.suptitle('The gauge wins day 1; the model wins the week (crossover day 2–4)',
                 fontsize=10.5, x=0.02, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f'{OUT}/persistence_crossover.png', bbox_inches='tight')
    plt.close(fig)


def fig_lobo() -> None:
    """Finding 3: Mill Ck held out of training -- skill inverts."""
    lobo_zarr = glob.glob(
        f'{RUNS}/lobo-MillCk_*/test/model_epoch*/test_results.zarr')[-1]
    gid = 'camels_11381500'
    o = obs_series(gid).loc['2022-10-01':'2023-06-30']
    gauged = sim_ts0(DET, gid).loc['2022-10-01':'2023-06-30']
    ungauged = sim_ts0(lobo_zarr, gid).loc['2022-10-01':'2023-06-30']
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ax.plot(o.index, o, color=INK, lw=1.1, label='observed')
    ax.plot(gauged.index, gauged, color=BLUE, lw=1.4, alpha=0.9,
            label='trained on this basin (NSE 0.73, both flood WYs)')
    ax.plot(ungauged.index, ungauged, color=PURPLE, lw=1.4, alpha=0.9,
            label='basin held out of training (NSE −0.74, both flood WYs)')
    ax.set_ylabel('mm/day')
    ax.margins(x=0.004)
    ax.legend(loc='upper right', frameon=False, fontsize=8)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.suptitle('Rain-driven Mill Ck, WY2023: held out of a snowmelt-trained model, '
                 'skill inverts', fontsize=10.5, x=0.02, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{OUT}/lobo_millck_inversion.png', bbox_inches='tight')
    plt.close(fig)


def fig_probabilistic() -> None:
    """Finding 4: CRPS by lead + interval coverage."""
    df = pd.read_csv(f'{OUT}/probabilistic_scores.csv')
    leads = sorted(df.lead.unique())
    cm = [df[(df.model == 'CMAL') & (df.lead == k)].CRPS.median() for k in leads]
    dt = [df[(df.model == 'deterministic') & (df.lead == k)].CRPS.median()
          for k in leads]
    cov = [df[(df.model == 'CMAL') & (df.lead == k)].coverage90.median()
           for k in leads]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.2))
    a1.plot(leads, dt, color=BLUE, lw=2, marker='o', ms=4, label='deterministic')
    a1.plot(leads, cm, color=PURPLE, lw=2, marker='o', ms=4, label='distributional')
    a1.set_ylabel('median CRPS (lower is better)')
    a1.set_xlabel('forecast lead (days)')
    a1.set_title('Sharper: 22–26% lower CRPS at every lead', fontsize=9.5)
    a1.legend(loc='center right', frameon=False, fontsize=8)
    a1.set_ylim(0, None)
    a2.axhspan(0.85, 0.95, color='#eceae6', zorder=0)
    a2.axhline(0.90, color='#8a8a86', lw=1, ls='--')
    a2.text(7.6, 0.905, 'nominal 90%', fontsize=7.5, color='#6b6b67', ha='right')
    a2.plot(leads, cov, color=PURPLE, lw=2, marker='o', ms=4)
    a2.set_ylim(0.5, 1.0)
    a2.set_xlabel('forecast lead (days)')
    a2.set_ylabel('coverage of 90% interval')
    a2.set_title('…but overconfident: 66–74% coverage', fontsize=9.5)
    fig.suptitle('The distributional head: sharper and uncalibrated',
                 fontsize=10.5, x=0.02, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{OUT}/crps_calibration.png', bbox_inches='tight')
    plt.close(fig)


def fig_nwm() -> None:
    """Finding 5: NWM comparison on flood years, grouped bars."""
    df = pd.read_csv(f'{OUT}/lstm_vs_nwm_flood_metrics.csv')
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True)
    for ax, (win, models) in zip(
        axes,
        [('WY2017', ['LSTM', 'NWM v2.1', 'NWM v3.0']),
         ('WY2023p', ['LSTM', 'NWM v3.0'])],
    ):
        sub = df[df.window == win]
        basins = [FOCUS[b] for b in FOCUS]
        x = np.arange(len(basins))
        w = 0.8 / len(models)
        colors = {'LSTM': BLUE, 'NWM v2.1': ORANGE2, 'NWM v3.0': ORANGE}
        for i, m in enumerate(models):
            vals = [float(sub[(sub.basin == b) & (sub.model == m)].NSE.iloc[0])
                    for b in FOCUS]
            ax.bar(x + (i - (len(models) - 1) / 2) * w, np.clip(vals, -0.55, None),
                   w * 0.92, color=colors[m], label=m,
                   hatch='//' if m == 'NWM v2.1' else None,
                   edgecolor='white', linewidth=0.8)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([b.split(' (')[0] for b in basins], rotation=18,
                           ha='right', fontsize=8)
        title = ('WY2017 (full year)' if win == 'WY2017'
                 else 'WY2023, Oct–Jan (v3.0 record ends)')
        ax.set_title(title, fontsize=9.5)
        ax.legend(loc='lower left', frameon=False, fontsize=8)
    axes[0].set_ylabel('NSE (clipped at −0.55)')
    fig.suptitle('The benchmark win depends on the benchmark: NWM v3.0 leads WY2017 '
                 '(3 of 5 basins); the LSTM leads WY2023 and Bear Ck',
                 fontsize=10.5, x=0.02, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{OUT}/lstm_vs_nwm_flood.png', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    fig_hydrographs()
    fig_persistence()
    fig_lobo()
    fig_probabilistic()
    fig_nwm()
    print('wrote 5 figures to', OUT)
