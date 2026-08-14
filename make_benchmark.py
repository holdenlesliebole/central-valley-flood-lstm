"""Benchmark the CA LSTM against the NOAA National Water Model retrospective (v2.1).
For each focus gauge: pull NWM CHRTOUT streamflow (m3/s) at its NHDPlus COMID,
daily-average, convert to mm/day via basin area, and score NWM vs the SAME observed
series the LSTM was scored on (test_results.zarr, lead 0). Outputs a comparison
table + overlay hydrographs + an NSE bar chart.
"""
import os, glob, time
import numpy as np, pandas as pd, xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import s3fs

FOCUS = {  # gauge -> (COMID, label)
    'camels_11264500': (21609533, 'Merced @ Happy Isles (snow)'),
    'camels_11266500': (21609641, 'Merced @ Pohono (snow)'),
    'camels_11230500': (17118323, 'Bear Ck (snow)'),
    'camels_11237500': (17116209, 'Pitman Ck (snow)'),
    'camels_11381500': (8019544,  'Mill Ck (rain)'),
}
rd = sorted(glob.glob(os.path.expanduser('~/Documents/Hydrology/runs/ca-28basin_*')))[-1]
figdir = os.path.join(rd, 'figures'); os.makedirs(figdir, exist_ok=True)

# areas (km2)
adf = pd.read_csv(os.path.expanduser('~/data/caravan-nc/attributes/camels/attributes_other_camels.csv'))
area = dict(zip(adf['gauge_id'], adf['area']))

def nse(o, s):
    m = ~np.isnan(o) & ~np.isnan(s)
    if m.sum() < 10: return np.nan
    o, s = o[m], s[m]; return 1 - np.sum((s-o)**2)/np.sum((o-o.mean())**2)
def kge(o, s):
    m = ~np.isnan(o) & ~np.isnan(s)
    if m.sum() < 10: return np.nan
    o, s = o[m], s[m]; r = np.corrcoef(o, s)[0,1]
    return 1 - np.sqrt((r-1)**2 + (s.std()/o.std()-1)**2 + (s.mean()/o.mean()-1)**2)

# ---- LSTM sim + obs (mm/day, lead 0) ----
res = xr.open_zarr(os.path.join(rd,'test','model_epoch015','test_results.zarr')).squeeze('freq')
ldates = pd.to_datetime(res['date'].values)

# ---- NWM pull (cache locally) ----
cache = os.path.expanduser('~/data/nwm_focus_2012_2014.nc')
if os.path.exists(cache):
    nwm = xr.open_dataset(cache)
    print("loaded NWM from cache")
else:
    print("opening NWM retrospective zarr ...")
    fs = s3fs.S3FileSystem(anon=True)
    ds = xr.open_zarr(s3fs.S3Map('noaa-nwm-retrospective-2-1-zarr-pds/chrtout.zarr', s3=fs), consolidated=True)
    comids = [c for c,_ in FOCUS.values()]
    t0=time.time()
    q = ds['streamflow'].sel(feature_id=comids).sel(time=slice('2012-01-01','2014-12-31'))
    q = q.resample(time='1D').mean().load()   # hourly -> daily mean (m3/s)
    print("pulled+daily NWM in %.0fs"%(time.time()-t0))
    nwm = q.to_dataset(name='streamflow_m3s')
    nwm.to_netcdf(cache)
ndates = pd.to_datetime(nwm['time'].values)

# ---- assemble comparison ----
rows = []
series = {}  # basin -> (dates, obs, lstm, nwm_mmday)
for gid,(comid,label) in FOCUS.items():
    b = res.sel(basin=gid)
    obs = pd.Series(b['streamflow_obs'].sel(time_step=0).values, index=ldates)
    lstm = pd.Series(b['streamflow_sim'].sel(time_step=0).values, index=ldates)
    qm3s = pd.Series(nwm['streamflow_m3s'].sel(feature_id=comid).values, index=ndates)
    nwm_mmday = qm3s * 86.4 / area[gid]                # m3/s -> mm/day
    # align to common dates
    df = pd.DataFrame({'obs':obs,'lstm':lstm,'nwm':nwm_mmday}).dropna(subset=['obs'])
    rows.append({'basin':label,
                 'LSTM_NSE':nse(df.obs.values,df.lstm.values),'LSTM_KGE':kge(df.obs.values,df.lstm.values),
                 'NWM_NSE':nse(df.obs.values,df.nwm.values),'NWM_KGE':kge(df.obs.values,df.nwm.values)})
    series[label]=df
tab = pd.DataFrame(rows)
print("\n=== LSTM vs NWM (test 2012-2014, common obs dates) ===")
print(tab.to_string(index=False, float_format=lambda x:f'{x:6.3f}'))
print(f"\nMEDIAN  LSTM NSE={tab.LSTM_NSE.median():.3f} KGE={tab.LSTM_KGE.median():.3f}"
      f"  |  NWM NSE={tab.NWM_NSE.median():.3f} KGE={tab.NWM_KGE.median():.3f}")
tab.to_csv(os.path.join(figdir,'lstm_vs_nwm_metrics.csv'), index=False)

# ---- Fig A: overlay hydrographs (obs, LSTM, NWM) ----
fig, axes = plt.subplots(5,1,figsize=(11,12),sharex=True)
for ax,(label,df) in zip(axes, series.items()):
    ax.plot(df.index, df.obs, color='0.25', lw=0.9, label='observed')
    ax.plot(df.index, df.lstm, color='tab:blue', lw=0.9, alpha=0.85, label='LSTM')
    ax.plot(df.index, df.nwm, color='tab:red', lw=0.9, alpha=0.8, label='NWM v2.1')
    r=tab[tab.basin==label].iloc[0]
    ax.text(0.01,0.92,f"{label}   LSTM NSE={r.LSTM_NSE:.2f} | NWM NSE={r.NWM_NSE:.2f}",
            transform=ax.transAxes, va='top', fontsize=10, fontweight='bold')
    ax.set_ylabel('Q (mm/d)'); ax.margins(x=0.005)
axes[0].legend(loc='upper right', fontsize=9, ncol=3)
axes[-1].xaxis.set_major_locator(mdates.YearLocator()); axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.suptitle('LSTM vs. NOAA National Water Model (v2.1 retrospective) vs. observed — test 2012–2014', fontsize=11)
fig.tight_layout(rect=[0,0,1,0.98]); fA=os.path.join(figdir,'lstm_vs_nwm_hydrographs.png'); fig.savefig(fA,dpi=140); plt.close(fig)

# ---- Fig B: NSE bar comparison ----
fig,ax=plt.subplots(figsize=(9,4.5))
x=np.arange(len(tab)); w=0.38
ax.bar(x-w/2, tab.LSTM_NSE, w, label='LSTM (this work)', color='tab:blue')
ax.bar(x+w/2, tab.NWM_NSE, w, label='NWM v2.1', color='tab:red')
ax.set_xticks(x); ax.set_xticklabels([b.split(' (')[0] for b in tab.basin], rotation=20, ha='right', fontsize=9)
ax.set_ylabel('NSE (test 2012–2014)'); ax.axhline(0,color='k',lw=0.6); ax.legend()
ax.set_title('Streamflow skill: CA-trained LSTM vs. National Water Model')
fig.tight_layout(); fB=os.path.join(figdir,'lstm_vs_nwm_nse_bars.png'); fig.savefig(fB,dpi=140); plt.close(fig)
print("\nwrote:", fA, fB)
