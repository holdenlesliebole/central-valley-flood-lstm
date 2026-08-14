"""Hydrograph + skill figures for the CA 28-basin model (test 2012-2014).
Reads test_results.zarr (lead times 0-7) and writes PNGs to the run's figures/.
Deterministic model (regression/MSE), so no predictive-interval calibration here;
we plot hydrographs, lead-time skill degradation, and obs-vs-sim scatter.
"""
import os, glob, sys
import numpy as np, pandas as pd, xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

rd = sorted(glob.glob(os.path.expanduser('~/Documents/Hydrology/runs/ca-28basin_*')))[-1]
z = os.path.join(rd, 'test', 'model_epoch015', 'test_results.zarr')
ds = xr.open_zarr(z).squeeze('freq')
figdir = os.path.join(rd, 'figures'); os.makedirs(figdir, exist_ok=True)

FOCUS = [
    ('camels_11264500', 'Merced R @ Happy Isles  (snow)'),
    ('camels_11266500', 'Merced R @ Pohono Bridge  (snow)'),
    ('camels_11230500', 'Bear Ck nr Lake Edison  (snow)'),
    ('camels_11237500', 'Pitman Ck  (snow)'),
    ('camels_11381500', 'Mill Ck nr Los Molinos  (rain)'),
]

def nse(o, s):
    m = ~np.isnan(o) & ~np.isnan(s)
    if m.sum() < 10: return np.nan
    o, s = o[m], s[m]
    return 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)

def kge(o, s):
    m = ~np.isnan(o) & ~np.isnan(s)
    if m.sum() < 10: return np.nan
    o, s = o[m], s[m]
    r = np.corrcoef(o, s)[0, 1]
    a = s.std() / o.std(); b = s.mean() / o.mean()
    return 1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2)

# ---- Figure 1: hydrographs, lead-0, 5 focus basins ----
dates = pd.to_datetime(ds['date'].values)
fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
for ax, (bid, name) in zip(axes, FOCUS):
    b = ds.sel(basin=bid)
    o = b['streamflow_obs'].sel(time_step=0).values
    s = b['streamflow_sim'].sel(time_step=0).values
    ax.plot(dates, o, color='0.25', lw=0.9, label='observed')
    ax.plot(dates, s, color='tab:blue', lw=0.9, alpha=0.85, label='simulated (lead 0)')
    ax.set_ylabel('Q (mm/d)')
    ax.text(0.01, 0.92, f"{name}   NSE={nse(o,s):.2f}  KGE={kge(o,s):.2f}",
            transform=ax.transAxes, va='top', fontsize=10, fontweight='bold')
    ax.margins(x=0.005)
axes[0].legend(loc='upper right', fontsize=9, ncol=2)
axes[-1].xaxis.set_major_locator(mdates.YearLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.suptitle('CA Sierra / Central-Valley LSTM — test period 2012–2014 (lead-0 daily streamflow)\n'
             '28-basin model; snowmelt freshet vs. observed. NB: 2012–14 was the onset of the CA drought (low peaks).',
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
f1 = os.path.join(figdir, 'hydrographs_focus5.png'); fig.savefig(f1, dpi=140); plt.close(fig)

# ---- Figure 2: lead-time skill degradation (0-7 days), focus basins + all-28 median ----
leads = ds['time_step'].values
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
for bid, name in FOCUS:
    b = ds.sel(basin=bid)
    nses = [nse(b['streamflow_obs'].sel(time_step=l).values, b['streamflow_sim'].sel(time_step=l).values) for l in leads]
    a1.plot(leads, nses, marker='o', ms=4, lw=1.2, label=name.split('  ')[0])
# all-28 median
alln = []
for l in leads:
    vals = [nse(ds['streamflow_obs'].sel(basin=bb, time_step=l).values,
                ds['streamflow_sim'].sel(basin=bb, time_step=l).values) for bb in ds['basin'].values]
    alln.append(np.nanmedian(vals))
a1.plot(leads, alln, marker='s', ms=5, lw=2.2, color='k', label='all 28 (median)')
a1.set_xlabel('forecast lead time (days)'); a1.set_ylabel('NSE'); a1.set_title('Skill vs. lead time')
a1.grid(alpha=0.3); a1.legend(fontsize=8)
# KGE version, all-28 median only + focus
for bid, name in FOCUS:
    b = ds.sel(basin=bid)
    kges = [kge(b['streamflow_obs'].sel(time_step=l).values, b['streamflow_sim'].sel(time_step=l).values) for l in leads]
    a2.plot(leads, kges, marker='o', ms=4, lw=1.2, label=name.split('  ')[0])
allk = []
for l in leads:
    vals = [kge(ds['streamflow_obs'].sel(basin=bb, time_step=l).values,
                ds['streamflow_sim'].sel(basin=bb, time_step=l).values) for bb in ds['basin'].values]
    allk.append(np.nanmedian(vals))
a2.plot(leads, allk, marker='s', ms=5, lw=2.2, color='k', label='all 28 (median)')
a2.set_xlabel('forecast lead time (days)'); a2.set_ylabel('KGE'); a2.set_title('KGE vs. lead time')
a2.grid(alpha=0.3)
fig.suptitle('Forecast skill degradation with lead time — CA 28-basin model, test 2012–2014', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
f2 = os.path.join(figdir, 'lead_time_skill.png'); fig.savefig(f2, dpi=140); plt.close(fig)

# ---- Figure 3: predicted-vs-observed scatter, lead-0, focus basins ----
fig, axes = plt.subplots(1, 5, figsize=(16, 3.4))
for ax, (bid, name) in zip(axes, FOCUS):
    b = ds.sel(basin=bid)
    o = b['streamflow_obs'].sel(time_step=0).values
    s = b['streamflow_sim'].sel(time_step=0).values
    m = ~np.isnan(o) & ~np.isnan(s)
    ax.scatter(o[m], s[m], s=6, alpha=0.35, color='tab:blue', edgecolors='none')
    hi = np.nanmax([o[m].max(), s[m].max()]) if m.sum() else 1
    ax.plot([0, hi], [0, hi], 'k--', lw=0.8)
    ax.set_title(name.split('  ')[0], fontsize=9)
    ax.set_xlabel('observed Q (mm/d)')
    ax.text(0.05, 0.93, f"NSE={nse(o,s):.2f}", transform=ax.transAxes, va='top', fontsize=9)
axes[0].set_ylabel('simulated Q (mm/d)')
fig.suptitle('Simulated vs. observed (lead 0), test 2012–2014 — 1:1 line dashed', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
f3 = os.path.join(figdir, 'scatter_focus5.png'); fig.savefig(f3, dpi=140); plt.close(fig)

print("wrote:")
for f in (f1, f2, f3): print("  ", f)
print("all-28 NSE by lead:", [f"{x:.2f}" for x in alln])
