"""Flow-conditional recalibration of the CMAL predictive distribution.

Why this exists
---------------
The distributional head is sharper than the deterministic one (22-26% CRPS,
METHODS 4.1d) but miscalibrated: 66-74% coverage at nominal 90%, with the
misses split by flow band (METHODS 4.1d Result 3). The PIT diagnosis implies
uniform variance inflation cannot repair coverage without destroying
sharpness, because the high-flow misses are displacement, not dispersion.
This script tests that implication directly.

Method
------
Cross-window conformal fitting: the correction is fit on ONE flood water year
and evaluated on the OTHER (both directions), so every reported number is
out-of-sample and the experiment directly tests whether a calibration learned
in one atmospheric-river winter transfers to another. (Fitting on the
2009-2011 validation years was the original design; the framework's inference
path yields an empty dataset for that period -- forecast-product inputs do
not assemble there -- so the flood years provide the split.) Days are
assigned to flow bands by the PREDICTED sample median (normalized per basin
by the fit window's mean predicted median), so the correction uses only
information available at forecast time.
Bands: quantiles [0, .5), [.5, .8), [.8, .95), [.95, 1] of the normalized
predictor, pooled across basins and leads.

Per band, a two-parameter affine map of the ensemble:
    x' = mu_b * (m + s_b * (x - m)),   m = per-day sample median
mu_b = median(obs / m) over band days in the fit window (multiplicative
median correction; days with m < 0.01 mm/day are excluded from fitting and
left uncorrected). s_b is set by bisection so the band's fit-window coverage
of the nominal 90% interval reaches 90% after the mu correction; s is capped
at 8 and the cap is reported if hit. Transformed samples are clipped at zero.
The composite result scores every test day with parameters fit on the other
window.

A GLOBAL comparator (one band, same functional form) is fit the same way, to
measure what uniform inflation alone achieves.

Outputs coverage and fair CRPS per lead on the test years for raw, global,
and flow-conditional variants, plus high/low-flow escape rates.

Valid-date convention -- FIXED 2026-08-19
-----------------------------------------
The zarr's `date` coordinate is the forecast ISSUE date and `time_step = k` is
the lead, so row (d, k) predicts flow on the VALID date d + k
(googlehydrology/evaluation/tester.py; closure-tested by
scripts/test_valid_date_alignment.py). Before 2026-08-19 this script built the
source-observation vector once on the issue dates and reused it at every lead,
so leads 1-7 were scored against flow k days too early; only lead 0 was right.
Every row now carries both `issue_date` and `valid_date = issue_date + lead`,
observations are read at the valid date, and the cross-fitting windows
(WY2017 / WY2023) are assigned by valid date. Rows whose valid date falls
outside both test water years -- the k-day overhang past each window end, whose
targets are training labels -- are dropped and the count is reported.

Usage:  python scripts/make_recalibration.py
Writes: outputs/figures/recalibration.csv (summary by variant x lead),
        outputs/figures/recalibration_by_basin.csv (per basin x lead x variant),
        outputs/figures/recalibration.png
"""

from __future__ import annotations

import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_probabilistic_scores import fair_crps  # noqa: E402

RUNS = os.path.expanduser('~/Documents/Side_projects/Hydrology/runs')
CMAL_RUN = f'{RUNS}/ca-28basin-cmal-flood-h128-nantol_1208_182446'
CARAVAN = os.path.expanduser('~/data/caravan-nc-extended/timeseries/netcdf/camels')
OUT = os.path.expanduser('~/Documents/Side_projects/Hydrology/outputs/figures')

NOMINAL = 0.90
BAND_EDGES = [0.0, 0.5, 0.8, 0.95, 1.0]   # quantiles of normalized predicted flow
S_CAP = 8.0
MIN_PRED = 0.01                            # mm/day; below this, no correction fit
THIN = 8                                   # ~940 of 7500 samples; keeps the flattened arrays ~2 GB
# Test water years, keyed on VALID date (not issue date).
WINDOWS_WY = {'WY2017': ('2016-10-01', '2017-09-30'),
              'WY2023': ('2022-10-01', '2023-09-30')}

INK, BLUE, PURPLE, GREEN = '#3a3a38', '#2f6bd8', '#8458d8', '#2e8a5c'


def load_period(zarr_path):
    ds = xr.open_zarr(zarr_path, consolidated=False).squeeze('freq')
    dates = pd.to_datetime(ds['date'].values)
    basins = [str(b) for b in ds.basin.values]
    return ds, dates, basins


def source_series(basin):
    """Full source-netCDF flow series for a basin, indexed by VALID date."""
    q = xr.open_dataset(f'{CARAVAN}/{basin}.nc')['streamflow'].to_series()
    q.index = pd.to_datetime(q.index)
    return q


def gather(zarr_path):
    """Flatten (basin, issue date, lead) -> rows with samples, obs, predictor.

    Observations are taken at valid_date = issue_date + lead; both dates are
    returned so no downstream row can exist without its date bookkeeping.
    """
    ds, dates, basins = load_period(zarr_path)
    sam, obs, med = [], [], []
    basin_ix, lead_ix, issue_ix, valid_ix = [], [], [], []
    for bi, b in enumerate(basins):
        q = source_series(b)
        for lead in range(ds.sizes['time_step']):
            vdates = dates + pd.Timedelta(days=lead)
            o = q.reindex(vdates).to_numpy().astype('float64')
            x = ds['streamflow_sim'].isel(
                basin=bi, time_step=lead
            ).values[:, ::THIN].astype('float64')
            m = np.nanmedian(x, axis=1)
            valid = np.isfinite(o) & np.isfinite(x).all(axis=1)
            sam.append(x[valid]); obs.append(o[valid]); med.append(m[valid])
            basin_ix.append(np.full(valid.sum(), bi))
            lead_ix.append(np.full(valid.sum(), lead))
            issue_ix.append(dates[valid].values)
            valid_ix.append(vdates[valid].values)
    return (np.concatenate(sam), np.concatenate(obs), np.concatenate(med),
            np.concatenate(basin_ix), np.concatenate(lead_ix),
            np.concatenate(issue_ix), np.concatenate(valid_ix), basins)


def coverage(sam, obs):
    lo = np.percentile(sam, 100 * (1 - NOMINAL) / 2, axis=1)
    hi = np.percentile(sam, 100 * (1 + NOMINAL) / 2, axis=1)
    return float(((obs >= lo) & (obs <= hi)).mean())


def apply_map(sam, med, mu, s):
    out = mu * (med[:, None] + s * (sam - med[:, None]))
    return np.clip(out, 0.0, None)


def fit_bands(sam, obs, med, norm, edges_q):
    """Fit (mu, s) per band on validation; returns band edges + params."""
    fit_mask = med >= MIN_PRED
    edges = np.quantile(norm[fit_mask], edges_q)
    edges[0], edges[-1] = -np.inf, np.inf
    params = []
    for k in range(len(edges) - 1):
        m_band = fit_mask & (norm >= edges[k]) & (norm < edges[k + 1])
        mu = float(np.median(obs[m_band] / med[m_band]))
        lo_s, hi_s = 0.25, S_CAP
        for _ in range(40):
            s = 0.5 * (lo_s + hi_s)
            cov = coverage(apply_map(sam[m_band], med[m_band], mu, s),
                           obs[m_band])
            if cov < NOMINAL:
                lo_s = s
            else:
                hi_s = s
        s = 0.5 * (lo_s + hi_s)
        cov = coverage(apply_map(sam[m_band], med[m_band], mu, s), obs[m_band])
        params.append({'band': k, 'mu': mu, 's': s, 'val_cov': cov,
                       'n_fit': int(m_band.sum()),
                       's_capped': bool(s > 0.98 * S_CAP)})
    return edges, params


def transform(sam, med, norm, edges, params):
    out = sam.copy()
    for p in params:
        k = p['band']
        m_band = (norm >= edges[k]) & (norm < edges[k + 1]) & (med >= MIN_PRED)
        out[m_band] = apply_map(sam[m_band], med[m_band], p['mu'], p['s'])
    return out


def main() -> None:
    test_zarr = f'{CMAL_RUN}/test/model_epoch016/test_results.zarr'
    ts, to, tm, tb, tl, t_issue, t_valid, basin_ids = gather(test_zarr)
    print(f'test rows before the valid-date window mask: {len(to)}')

    # Windows are assigned by VALID date; the lead overhang past each window end
    # verifies inside the next training period and is dropped.
    windows = {name: (t_valid >= np.datetime64(s)) & (t_valid <= np.datetime64(e))
               for name, (s, e) in WINDOWS_WY.items()}
    keep = np.zeros(len(to), dtype=bool)
    for m in windows.values():
        keep |= m
    print(f'dropped {int((~keep).sum())} rows with valid_date outside both test '
          f'water years (lead overhang into training targets)')
    ts, to, tm = ts[keep], to[keep], tm[keep]
    tb, tl = tb[keep], tl[keep]
    t_issue, t_valid = t_issue[keep], t_valid[keep]
    windows = {k: v[keep] for k, v in windows.items()}
    print(f'test rows: {len(to)}')
    print({k: int(m.sum()) for k, m in windows.items()})

    # Composite: every row transformed with parameters fit on the OTHER window.
    out = {'raw': ts.copy(), 'global': ts.copy(), 'flow-conditional': ts.copy()}
    for fit_name, fit_m in windows.items():
        eval_m = ~fit_m
        # Per-basin normalizer from the FIT window's predicted medians only.
        basins = np.unique(tb)
        norm_ref = {b: np.mean(tm[fit_m & (tb == b) & (tm >= MIN_PRED)])
                    for b in basins}
        fnorm = tm[fit_m] / np.array([norm_ref[b] for b in tb[fit_m]])
        enorm = tm[eval_m] / np.array([norm_ref[b] for b in tb[eval_m]])

        edges_fc, params_fc = fit_bands(ts[fit_m], to[fit_m], tm[fit_m],
                                        fnorm, BAND_EDGES)
        edges_gl, params_gl = fit_bands(ts[fit_m], to[fit_m], tm[fit_m],
                                        fnorm, [0.0, 1.0])
        print(f'\nfit on {fit_name}, flow-conditional:')
        print(pd.DataFrame(params_fc).to_string(index=False))
        print(f'fit on {fit_name}, global:')
        print(pd.DataFrame(params_gl).to_string(index=False))
        out['global'][eval_m] = transform(
            ts[eval_m], tm[eval_m], enorm, edges_gl, params_gl)
        out['flow-conditional'][eval_m] = transform(
            ts[eval_m], tm[eval_m], enorm, edges_fc, params_fc)

    variants = out

    def dspan(m):
        """issue/valid date span of a row subset, as an explicit bookkeeping stamp."""
        return {
            'issue_date_first': str(pd.Timestamp(t_issue[m].min()).date()),
            'issue_date_last': str(pd.Timestamp(t_issue[m].max()).date()),
            'valid_date_first': str(pd.Timestamp(t_valid[m].min()).date()),
            'valid_date_last': str(pd.Timestamp(t_valid[m].max()).date()),
        }

    rows = []
    ter = np.quantile(to, [2 / 3])
    for name, x in variants.items():
        for lead in sorted(np.unique(tl)):
            m = tl == lead
            rows.append({
                'variant': name, 'lead': int(lead), 'n': int(m.sum()),
                'coverage90': coverage(x[m], to[m]),
                'crps': float(np.mean(fair_crps(x[m], to[m]))),
                **dspan(m),
            })
        hi = to > ter[0]
        lo_esc = float((to[hi] < np.percentile(x[hi], 5, axis=1)).mean())
        hi_esc = float((to[hi] > np.percentile(x[hi], 95, axis=1)).mean())
        rows.append({'variant': name, 'lead': -1, 'n': int(hi.sum()),
                     'coverage90': coverage(x[hi], to[hi]),
                     'crps': float(np.mean(fair_crps(x[hi], to[hi]))),
                     **dspan(hi),
                     'note': f'top-tercile days; esc_above={hi_esc:.3f}, '
                             f'esc_below={lo_esc:.3f}'})
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(f'{OUT}/recalibration.csv', index=False)

    # Long-form companion: one row per basin x lead x variant, each stamped with
    # the issue- and valid-date span it covers (review action item, 2026-08-19).
    brows = []
    for name, x in variants.items():
        for bi in sorted(np.unique(tb)):
            for lead in sorted(np.unique(tl)):
                m = (tb == bi) & (tl == lead)
                if not m.any():
                    continue
                brows.append({
                    'basin': basin_ids[bi], 'variant': name, 'lead': int(lead),
                    'n': int(m.sum()), **dspan(m),
                    'coverage90': coverage(x[m], to[m]),
                    'crps': float(np.mean(fair_crps(x[m], to[m]))),
                })
    pd.DataFrame(brows).to_csv(f'{OUT}/recalibration_by_basin.csv', index=False)

    piv_c = df[df.lead >= 0].pivot(index='lead', columns='variant',
                                   values='coverage90')
    piv_s = df[df.lead >= 0].pivot(index='lead', columns='variant',
                                   values='crps')
    print('\ncoverage by lead (test):')
    print(piv_c.to_string(float_format=lambda v: f'{v:.3f}'))
    print('\nfair CRPS by lead (test, mm/day):')
    print(piv_s.to_string(float_format=lambda v: f'{v:.3f}'))
    print('\ntop-tercile rows:')
    print(df[df.lead == -1].to_string(index=False))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.3))
    colors = {'raw': PURPLE, 'global': BLUE, 'flow-conditional': GREEN}
    for name in variants:
        a1.plot(piv_c.index, piv_c[name], color=colors[name], lw=2,
                marker='o', ms=4, label=name)
        a2.plot(piv_s.index, piv_s[name], color=colors[name], lw=2,
                marker='o', ms=4, label=name)
    a1.axhspan(0.85, 0.95, color='#eceae6', zorder=0)
    a1.axhline(NOMINAL, color='#8a8a86', lw=1, ls='--')
    a1.set_ylim(0.5, 1.0)
    a1.set_xlabel('forecast lead (days)')
    a1.set_ylabel('coverage of nominal 90% interval')
    a1.legend(loc='lower right', frameon=False, fontsize=8)
    a2.set_xlabel('forecast lead (days)')
    a2.set_ylabel('fair CRPS (mm/day)')
    a2.set_ylim(0, None)
    fig.suptitle('Coverage and fair CRPS by lead, flood test years: raw '
                 'ensemble vs affine recalibration, each window scored with '
                 'parameters fit on the other\n(scored at valid date = issue '
                 'date + lead)', fontsize=10.5, x=0.02, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f'{OUT}/recalibration.png', dpi=150, bbox_inches='tight')
    print(f'\nwrote {OUT}/recalibration.csv, recalibration_by_basin.csv '
          f'and recalibration.png')


if __name__ == '__main__':
    main()
