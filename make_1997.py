"""Held-out Jan-1997 flood analysis for the ca-28basin-1997 model.
Test period 1993-1999 was held out of training (train 2002-2014). Zooms on water
year 1997 (Oct 1996-Sep 1997) to assess peak capture vs. NWM v2.1, with peak
magnitude/timing metrics. Run AFTER `run infer` on the 1997 model.
"""
import os, glob
import numpy as np, pandas as pd, xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

FOCUS = {
    'camels_11264500': (21609533, 'Merced @ Happy Isles (snow)'),
    'camels_11266500': (21609641, 'Merced @ Pohono (snow)'),
    'camels_11230500': (17118323, 'Bear Ck (snow)'),
    'camels_11237500': (17116209, 'Pitman Ck (snow)'),
    'camels_11381500': (8019544,  'Mill Ck (rain)'),
}
rd = sorted(glob.glob(os.path.expanduser('~/Documents/Side_projects/Hydrology/runs/ca-28basin-1997_*')))[-1]
figdir = os.path.join(rd, 'figures'); os.makedirs(figdir, exist_ok=True)
adf = pd.read_csv(os.path.expanduser('~/data/caravan-nc/attributes/camels/attributes_other_camels.csv'))
area = dict(zip(adf['gauge_id'], adf['area']))

def nse(o,s):
    m=~np.isnan(o)&~np.isnan(s)
    if m.sum()<5: return np.nan
    o,s=o[m],s[m]; return 1-np.sum((s-o)**2)/np.sum((o-o.mean())**2)

res = xr.open_zarr(os.path.join(rd,'test','model_epoch015','test_results.zarr')).squeeze('freq')
ld = pd.to_datetime(res['date'].values)
nwm = xr.open_dataset(os.path.expanduser('~/data/nwm_focus_1995_1999.nc'))
nd = pd.to_datetime(nwm['time'].values)

WY = (pd.Timestamp('1996-10-01'), pd.Timestamp('1997-09-30'))   # water year 1997
EV = (pd.Timestamp('1996-11-15'), pd.Timestamp('1997-04-30'))   # flood window for plotting

rows=[]; series={}
for gid,(comid,label) in FOCUS.items():
    b=res.sel(basin=gid)
    obs=pd.Series(b['streamflow_obs'].sel(time_step=0).values, index=ld)
    lstm=pd.Series(b['streamflow_sim'].sel(time_step=0).values, index=ld)
    nwm_mm=pd.Series(nwm['streamflow_m3s'].sel(feature_id=comid).values*86.4/area[gid], index=nd)
    df=pd.DataFrame({'obs':obs,'lstm':lstm,'nwm':nwm_mm})
    wy=df.loc[WY[0]:WY[1]]
    o=wy['obs']
    if o.notna().sum()<5:
        continue
    opk_d=o.idxmax(); opk=o.max()
    def peakstats(col):
        s=wy[col]
        if s.notna().sum()<5: return np.nan,np.nan,np.nan
        pk=s.max(); pkd=s.idxmax()
        return pk, 100*(pk-opk)/opk, (pkd-opk_d).days
    lpk,lerr,ltim=peakstats('lstm'); npk,nerr,ntim=peakstats('nwm')
    rows.append({'basin':label,'obs_peak_mmd':round(opk,2),'obs_peak_date':str(opk_d.date()),
                 'LSTM_WY97_NSE':round(nse(o.values,wy['lstm'].values),3),'LSTM_peak_err%':round(lerr,0),'LSTM_peak_lag_d':ltim,
                 'NWM_WY97_NSE':round(nse(o.values,wy['nwm'].values),3),'NWM_peak_err%':round(nerr,0),'NWM_peak_lag_d':ntim})
    series[label]=df.loc[EV[0]:EV[1]]
tab=pd.DataFrame(rows)
pd.set_option('display.width',200)
print("=== Held-out Water Year 1997 (Jan-1997 flood) — peak capture ===")
print(tab.to_string(index=False))
tab.to_csv(os.path.join(figdir,'flood1997_metrics.csv'), index=False)

# zoomed hydrographs over the flood window
fig,axes=plt.subplots(5,1,figsize=(11,12),sharex=True)
for ax,(label,df) in zip(axes,series.items()):
    ax.plot(df.index,df['obs'],color='0.25',lw=1.1,label='observed')
    ax.plot(df.index,df['lstm'],color='tab:blue',lw=1.1,alpha=0.85,label='LSTM (held out)')
    ax.plot(df.index,df['nwm'],color='tab:red',lw=1.1,alpha=0.8,label='NWM v2.1')
    ax.axvline(pd.Timestamp('1997-01-02'),color='0.6',ls=':',lw=1)
    r=tab[tab.basin==label]
    if len(r):
        r=r.iloc[0]
        ax.text(0.01,0.93,f"{label}   LSTM NSE={r['LSTM_WY97_NSE']:.2f} (peak {r['LSTM_peak_err%']:+.0f}%) | "
                f"NWM NSE={r['NWM_WY97_NSE']:.2f} (peak {r['NWM_peak_err%']:+.0f}%)",
                transform=ax.transAxes,va='top',fontsize=9.5,fontweight='bold')
    ax.set_ylabel('Q (mm/d)'); ax.margins(x=0.005)
axes[0].legend(loc='upper right',fontsize=9,ncol=3)
axes[-1].xaxis.set_major_locator(mdates.MonthLocator()); axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
fig.suptitle('Held-out January 1997 flood (water year 1997) — LSTM vs NWM vs observed\n'
             'Model never saw 1993–1999; dotted line = ~2 Jan 1997 flood peak', fontsize=11)
fig.tight_layout(rect=[0,0,1,0.97]); f=os.path.join(figdir,'flood1997_hydrographs.png'); fig.savefig(f,dpi=140); plt.close(fig)
print("wrote", f)
