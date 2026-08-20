"""Does the LSTM's advantage over gauge persistence survive inside storm bins?

Why this exists
---------------
METHODS 4.2 establishes the crossover as a whole-period, cohort-median NSE
statement: yesterday's gauge wins at lead 1, the LSTM overtakes at lead 2 (plain)
or lead 3-4 (damped), and both nulls collapse at long leads. The 2026-08-19
adversarial review sharpened that to per-basin win counts. Neither is conditioned
on storm size, and an operator does not care about the median day -- they care
about the P80-P95 and >P95 storm days, which is exactly where
`make_storm_stratified_skill.py` shows the LSTM's bias breaking down (-0.30 and
-0.61 at lead 3 after the valid-date fix). Persistence has its own failure mode on
those days -- a storm day is by construction poorly predicted by the day the storm
had not yet arrived -- so which null wins on the tail is not deducible from the
aggregate. This script measures it.

Conventions (all inherited, none invented)
------------------------------------------
Valid date. The zarr's `date` is the ISSUE date and `time_step = k` the lead, so
row (d, k) is a forecast for valid date v = d + k
(scripts/test_valid_date_alignment.py closure-tests this). Observation,
precipitation percentile, storm class and the water-year window mask are all taken
at v, exactly as in the corrected `make_storm_stratified_skill.py`.

Baselines, matched by lead (METHODS 4.2, `make_baselines.py::persistence_by_lead`):
  plain   pers(d, k)   = q(v - k) = q(d)   -- the last gauge reading at issue time
  damped  damp(d, k)   = clim(v) + alpha**k * (q(d) - clim(d)), clipped at 0
with `clim` the 31-day circularly-smoothed day-of-year climatology fitted on the
flood-split TRAINING periods only, and `alpha` the lag-1 autocorrelation of
training-period anomalies about that climatology. The climatology, alpha fit and
damped form are imported from `make_baselines` rather than re-implemented, and the
per-basin alpha and per-lead NSE are closure-checked against the committed
`persistence_by_lead.csv` before anything is reported (see `closure_checks`).

Note that v - k = d identically, so plain persistence at lead k is the observation
on the issue date. Both models therefore know everything through day d and nothing
after it: the LSTM has day-d weather analysis plus forecast forcing, persistence
has the day-d gauge. The standing asymmetry still applies and must travel with
these numbers -- the LSTM never ingests observed discharge at all, while a real
operator has the gauge.

Leads. 1-7 only. time_step 0 is a same-day nowcast with no exact persistence
analogue (METHODS 4.2); including it would require inventing a convention.

Metrics per (storm bin, lead), pooled over the 22-basin cohort on a row set where
LSTM, plain and damped are ALL finite, so the three models are scored on identical
days:
  n, mean_obs, MAE for each model, MAE skill score 1 - MAE_lstm/MAE_baseline,
  relative bias mean(sim-obs)/mean(obs) for each model, and the win rate
  = fraction of basin-days with |err_lstm| < |err_baseline|.
Within-bin NSE is deliberately NOT the headline: conditioning on precipitation
truncates the observed variance, so a bin NSE measures the bin's own spread rather
than model skill. Pooled NSE is carried as footnote columns only.

Dry-bin caveat: MAE ratios on near-zero flows are unstable. `mean_obs` and the raw
MAE columns are reported so a skill score driven by a tiny denominator is visible;
the script prints an explicit flag for any bin whose baseline MAE is below
MAE_FLOOR_FLAG.

Usage:  source ~/opt/anaconda3/etc/profile.d/conda.sh && conda activate googlehydrology
        cd ~/Documents/Side_projects/Hydrology
        python scripts/make_storm_stratified_persistence.py
Writes: outputs/figures/storm_stratified_persistence.csv
        outputs/figures/storm_stratified_persistence.png
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

from googlehydrology.evaluation.metrics import nse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_baselines import (  # noqa: E402
    SMOOTH_WINDOW, TEST_PERIODS, TRAIN_PERIODS, climatology_on_index,
    day_of_year_climatology, select_periods,
)
from make_storm_stratified_skill import (  # noqa: E402
    BIN_ORDER, CARAVAN, DRY_THRESHOLD_MM, MIN_VALID_OBS_DAYS, OUT_DIR,
    ZARR_PATH, assign_bin, fetch_precip_cache, in_windows, wet_day_percentile,
)

LEADS = list(range(1, 8))
MAE_FLOOR_FLAG = 0.05        # mm/day; below this a skill ratio is not trustworthy
PERS_BY_LEAD_CSV = f'{OUT_DIR}/persistence_by_lead.csv'

BLUE, PURPLE, GREEN, INK = '#2f6bd8', '#8458d8', '#2e8a5c', '#3a3a38'

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150, 'font.size': 9,
    'axes.grid': True, 'grid.color': '#e6e6e2', 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#c9c9c4', 'axes.linewidth': 0.8,
})


def fit_climatology_and_alpha(q: pd.Series) -> tuple[pd.Series, float]:
    """Smoothed training climatology and lag-1 anomaly autocorrelation.

    Line-for-line the fit in make_baselines.persistence_by_lead, so the
    parameters reconcile with the committed persistence_by_lead.csv.
    """
    train = select_periods(q, TRAIN_PERIODS)
    clim_sm = day_of_year_climatology(train.dropna(), SMOOTH_WINDOW)
    tr_clim = climatology_on_index(clim_sm, train.index)
    anom = train.to_numpy() - tr_clim
    a0, a1 = anom[:-1], anom[1:]
    m = ~np.isnan(a0) & ~np.isnan(a1)
    alpha = float(np.corrcoef(a0[m], a1[m])[0, 1]) if m.sum() > 10 else 0.0
    alpha = float(np.clip(alpha, 0.0, 1.0)) if np.isfinite(alpha) else 0.0
    return clim_sm, alpha


def build_table() -> tuple[pd.DataFrame, dict]:
    """One row per (basin, issue_date, lead) with obs, LSTM, both baselines and
    the storm class -- everything evaluated at valid_date = issue_date + lead."""
    res = xr.open_zarr(ZARR_PATH, consolidated=False).squeeze('freq')
    issue_dates = pd.to_datetime(res.date.values)
    issue_window_mask = in_windows(issue_dates)
    precip_all = fetch_precip_cache()

    rows, cohort, alphas = [], [], {}
    for basin in [str(b) for b in res.basin.values]:
        q = xr.open_dataset(f'{CARAVAN}/{basin}.nc')['streamflow'].to_series()
        q.index = pd.to_datetime(q.index)

        # Same cohort rule and same lead-0 axis as make_storm_stratified_skill.
        if int((q.reindex(issue_dates).notna().to_numpy()
                & issue_window_mask).sum()) < MIN_VALID_OBS_DAYS:
            continue
        cohort.append(basin)

        clim_sm, alpha = fit_climatology_and_alpha(q)
        alphas[basin] = alpha

        precip_full = precip_all.sel(basin=basin).to_series()
        precip_full.index = pd.to_datetime(precip_full.index)
        storm_bin_full = assign_bin(precip_full < DRY_THRESHOLD_MM,
                                    wet_day_percentile(precip_full))

        b = res.sel(basin=basin)
        for lead in LEADS:
            valid_dates = issue_dates + pd.Timedelta(days=lead)
            # v - lead == issue date, so plain persistence is the issue-day gauge.
            pers = q.reindex(issue_dates).to_numpy()
            clim_v = climatology_on_index(clim_sm, valid_dates)
            clim_d = climatology_on_index(clim_sm, issue_dates)
            damped = np.clip(clim_v + alpha ** lead * (pers - clim_d), 0, None)
            df = pd.DataFrame({
                'basin': basin,
                'issue_date': issue_dates,
                'lead': lead,
                'valid_date': valid_dates,
                'obs': q.reindex(valid_dates).to_numpy(),
                'lstm': b['streamflow_sim'].sel(time_step=lead).values,
                'persistence': pers,
                'damped': damped,
                'storm_bin': storm_bin_full.reindex(valid_dates).to_numpy(),
            })
            rows.append(df[in_windows(valid_dates)])

    print(f'Cohort: {len(cohort)} basins.')
    out = pd.concat(rows, ignore_index=True)
    # Common row set: all three models finite, so nothing is scored on days a
    # competitor was dropped from (guardrail Rule 17 -- the baseline must be a
    # real lower bound, not a differently-sampled one).
    before = len(out)
    out = out.dropna(subset=['obs', 'lstm', 'persistence', 'damped'])
    print(f'Rows: {len(out)} of {before} with all three models and obs finite.')
    return out, {'cohort': cohort, 'alphas': alphas}


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    groups = [(b, lead) for b in BIN_ORDER + ['ALL'] for lead in LEADS]
    for storm_bin, lead in groups:
        sub = df[df['lead'] == lead]
        if storm_bin != 'ALL':
            sub = sub[sub['storm_bin'] == storm_bin]
        n = len(sub)
        if n == 0:
            out.append({'storm_bin': storm_bin, 'lead': lead, 'n_basin_days': 0})
            continue
        o = sub['obs'].to_numpy()
        obs_mean = o.mean()
        rec = {'storm_bin': storm_bin, 'lead': lead, 'n_basin_days': n,
               'mean_obs': obs_mean,
               'valid_date_first': str(sub['valid_date'].min().date()),
               'valid_date_last': str(sub['valid_date'].max().date())}
        err = {}
        for name in ('lstm', 'persistence', 'damped'):
            e = sub[name].to_numpy() - o
            err[name] = e
            rec[f'mae_{name}'] = float(np.abs(e).mean())
            rec[f'rel_bias_{name}'] = float(e.mean() / obs_mean)
            rec[f'pooled_nse_{name}'] = float(nse(
                xr.DataArray(o, dims='d'), xr.DataArray(sub[name].to_numpy(), dims='d')))
        for base in ('persistence', 'damped'):
            rec[f'mae_skill_vs_{base}'] = float(
                1.0 - rec['mae_lstm'] / rec[f'mae_{base}'])
            rec[f'winrate_vs_{base}'] = float(
                (np.abs(err['lstm']) < np.abs(err[base])).mean())
        out.append(rec)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Closure checks. Nothing is written until these pass or are reported.
# --------------------------------------------------------------------------

def closure_checks(df: pd.DataFrame, meta: dict) -> None:
    print('\n' + '=' * 70)
    print('CLOSURE CHECKS')
    print('=' * 70)

    # (0) alpha reconciles with the committed persistence_by_lead.csv.
    ref = pd.read_csv(PERS_BY_LEAD_CSV).set_index('basin')
    da = max(abs(meta['alphas'][b] - float(ref.loc[b, 'alpha']))
             for b in meta['cohort'])
    print(f'(0) alpha vs persistence_by_lead.csv: max abs diff {da:.3e}  '
          f'{"PASS" if da < 1e-12 else "FAIL"}')

    # (1) Reproduce persistence_by_lead.csv NSE using ITS masking (obs-only, no
    #     LSTM-availability requirement), which proves the baseline construction
    #     is the project's own and not a look-alike.
    worst_p = worst_d = 0.0
    for basin in meta['cohort']:
        q = xr.open_dataset(f'{CARAVAN}/{basin}.nc')['streamflow'].to_series()
        q.index = pd.to_datetime(q.index)
        clim_sm, alpha = fit_climatology_and_alpha(q)
        tgt = select_periods(q, TEST_PERIODS)
        o = tgt.to_numpy()
        idx = tgt.index
        clim_v = climatology_on_index(clim_sm, idx)
        for lead in LEADS:
            prev = q.reindex(idx - pd.Timedelta(days=lead)).to_numpy()
            clim_p = climatology_on_index(clim_sm, idx - pd.Timedelta(days=lead))
            damp = np.clip(clim_v + alpha ** lead * (prev - clim_p), 0, None)
            for sim, col, store in ((prev, str(lead), 'p'),
                                    (damp, f'damped_{lead}', 'd')):
                m = np.isfinite(o) & np.isfinite(sim)
                got = float(nse(xr.DataArray(o[m], dims='d'),
                                xr.DataArray(sim[m], dims='d')))
                diff = abs(got - float(ref.loc[basin, col]))
                if store == 'p':
                    worst_p = max(worst_p, diff)
                else:
                    worst_d = max(worst_d, diff)
    print(f'(1) per-basin per-lead NSE vs persistence_by_lead.csv: '
          f'plain max |diff| {worst_p:.3e}, damped max |diff| {worst_d:.3e}  '
          f'{"PASS" if max(worst_p, worst_d) < 1e-9 else "FAIL"}')

    # (2) Aggregate direction: per-basin NSE win counts vs both nulls, on the
    #     common row set. Expected shape from the review: LSTM loses at lead 1
    #     on most basins, wins on all 22 by lead 4.
    print('(2) per-basin NSE win counts on the common row set '
          '(expect: lead 1 mostly losses, all 22 by lead 4)')
    print(f'    {"lead":>4}{"LSTM > plain":>14}{"LSTM > damped":>15}')
    for lead in LEADS:
        s = df[df['lead'] == lead]
        wp = wd = 0
        for basin, g in s.groupby('basin'):
            o = xr.DataArray(g['obs'].to_numpy(), dims='d')
            n_l = float(nse(o, xr.DataArray(g['lstm'].to_numpy(), dims='d')))
            n_p = float(nse(o, xr.DataArray(g['persistence'].to_numpy(), dims='d')))
            n_d = float(nse(o, xr.DataArray(g['damped'].to_numpy(), dims='d')))
            wp += n_l > n_p
            wd += n_l > n_d
        n_b = s['basin'].nunique()
        print(f'    {lead:>4}{f"{wp}/{n_b}":>14}{f"{wd}/{n_b}":>15}')

    # (3) Hand check: one basin, one lead, one bin, first few days, recomputed
    #     from the raw series with no shared helper.
    basin, lead, bin_name = 'camels_11381500', 3, '>P95'
    g = df[(df.basin == basin) & (df.lead == lead) &
           (df.storm_bin == bin_name)].head(5)
    q = xr.open_dataset(f'{CARAVAN}/{basin}.nc')['streamflow'].to_series()
    q.index = pd.to_datetime(q.index)
    res = xr.open_zarr(ZARR_PATH, consolidated=False).squeeze('freq')
    zi = pd.to_datetime(res.date.values)
    print(f'(3) hand check -- {basin}, lead {lead}, bin "{bin_name}", first 5 rows')
    print(f'    {"issue":>12}{"valid":>12}{"obs":>9}{"obs_hand":>10}'
          f'{"lstm":>9}{"lstm_hand":>11}{"pers":>9}{"pers_hand":>11}')
    ok = True
    for _, r in g.iterrows():
        obs_h = float(q.loc[r.valid_date])
        lstm_h = float(res['streamflow_sim'].sel(basin=basin, time_step=lead)
                       .values[int(np.where(zi == r.issue_date)[0][0])])
        pers_h = float(q.loc[r.issue_date])
        ok &= (abs(obs_h - r.obs) < 1e-5 and abs(lstm_h - r.lstm) < 1e-5
               and abs(pers_h - r.persistence) < 1e-5)
        print(f'    {str(r.issue_date.date()):>12}{str(r.valid_date.date()):>12}'
              f'{r.obs:>9.3f}{obs_h:>10.3f}{r.lstm:>9.3f}{lstm_h:>11.3f}'
              f'{r.persistence:>9.3f}{pers_h:>11.3f}')
    print(f'    {"PASS" if ok else "FAIL"}: valid_date = issue_date + {lead}, '
          f'persistence = obs on the issue date')

    # (4) Row counts reconcile with the corrected storm_stratified_skill.csv at
    #     lead 3. That table has no persistence-availability requirement, so this
    #     script's counts must be <= its counts, and equal wherever the gauge was
    #     available on the issue date.
    skill = pd.read_csv(f'{OUT_DIR}/storm_stratified_skill.csv', comment='#')
    s3 = skill[skill.lead == 3].set_index('storm_bin')['n_basin_days']
    m3 = df[df.lead == 3].groupby('storm_bin').size()
    print('(4) lead-3 row counts vs storm_stratified_skill.csv')
    print(f'    {"bin":<12}{"skill csv":>11}{"here":>8}{"delta":>8}')
    for b in BIN_ORDER:
        print(f'    {b:<12}{int(s3[b]):>11}{int(m3.get(b, 0)):>8}'
              f'{int(m3.get(b, 0)) - int(s3[b]):>8}')
    print(f'    total{int(s3.sum()):>18}{int(m3.sum()):>8}'
          f'{int(m3.sum()) - int(s3.sum()):>8}')


def make_figure(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    for ax, base, title in zip(
            axes, ('persistence', 'damped'),
            ('vs plain persistence', 'vs damped persistence')):
        for b, c in zip(['dry', 'P50-P80', 'P80-P95', '>P95'],
                        ['#b9b9b4', GREEN, BLUE, PURPLE]):
            s = summary[summary.storm_bin == b].sort_values('lead')
            ax.plot(s['lead'], s[f'mae_skill_vs_{base}'], marker='o', ms=4,
                    lw=2, color=c, label=b)
        ax.axhline(0, color=INK, lw=0.9)
        ax.set_xlabel('forecast lead (days)')
        ax.set_title(title, fontsize=9.5)
    axes[0].set_ylabel('MAE skill score\n1 - MAE$_{LSTM}$/MAE$_{baseline}$')
    axes[1].legend(loc='lower right', frameon=False, fontsize=8, title='storm bin',
                   title_fontsize=8)
    fig.suptitle('LSTM vs gauge persistence by storm size and lead, WY2017+WY2023, '
                 'cohort n=22\n(above zero = LSTM better; valid-date aligned, '
                 'matched lead)', fontsize=10.5, x=0.02, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    path = f'{OUT_DIR}/storm_stratified_persistence.png'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {path}')


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df, meta = build_table()
    closure_checks(df, meta)
    summary = summarize(df)

    out_csv = f'{OUT_DIR}/storm_stratified_persistence.csv'
    with open(out_csv, 'w') as f:
        f.write(
            '# LSTM vs gauge persistence by storm-size bin and lead, '
            'WY2017+WY2023, 22-basin cohort.\n'
            '# VALID-DATE SCORING: a forecast issued at d for lead k is scored at '
            'valid_date v = d + k; obs, precipitation percentile, storm class and '
            'the water-year mask are all taken at v.\n'
            '# Matched-lead baselines (METHODS 4.2 / make_baselines.py): '
            'persistence = obs at v-k = obs on the issue date; damped = '
            'clim(v) + alpha**k * (obs(d) - clim(d)), clipped at 0, with the '
            '31-day smoothed training day-of-year climatology and alpha the '
            'training lag-1 anomaly autocorrelation.\n'
            '# Rows require obs, LSTM, plain and damped all finite, so the three '
            'models are scored on identical days.\n'
            '# mae_skill_vs_X = 1 - mae_lstm/mae_X (positive = LSTM better); '
            'winrate_vs_X = fraction of basin-days with |err_lstm| < |err_X|; '
            'rel_bias = mean(sim-obs)/mean(obs).\n'
            '# pooled_nse_* are FOOTNOTE columns only: within-bin NSE is computed '
            'on a precipitation-truncated variance and is not a skill measure.\n'
            '# Leads 1-7 only; time_step 0 is a nowcast with no exact persistence '
            'analogue (METHODS 4.2). storm_bin "ALL" pools every bin.\n'
            '# The standing asymmetry: the LSTM never ingests observed discharge; '
            'persistence uses the gauge, which a real operator also has.\n'
        )
        summary.to_csv(f, index=False)
    print(f'\nWrote {out_csv}')

    # Only the three raw MAE columns -- NOT mae_skill_vs_*, which are skill
    # scores and are legitimately negative.
    mae_cols = ['mae_lstm', 'mae_persistence', 'mae_damped']
    thin = summary[summary[mae_cols].min(axis=1) < MAE_FLOOR_FLAG]
    if len(thin):
        print(f'\nFLAG: {len(thin)} bin/lead cells have a model MAE below '
              f'{MAE_FLOOR_FLAG} mm/day; their skill ratios rest on a tiny '
              f'denominator:')
        print(thin[['storm_bin', 'lead', 'mean_obs', 'mae_lstm',
                    'mae_persistence', 'mae_damped']].to_string(index=False))
    else:
        print(f'\nNo bin/lead cell has a model MAE below {MAE_FLOOR_FLAG} mm/day; '
              f'no skill score rests on a near-zero denominator.')

    make_figure(summary)

    pd.set_option('display.width', 220)
    show = ['storm_bin', 'lead', 'n_basin_days', 'mean_obs', 'mae_lstm',
            'mae_persistence', 'mae_damped', 'mae_skill_vs_persistence',
            'mae_skill_vs_damped', 'winrate_vs_persistence', 'winrate_vs_damped']
    print('\n=== LSTM vs persistence by storm bin and lead (n=22 basins) ===')
    print(summary[show].to_string(index=False,
                                  float_format=lambda v: f'{v:.3f}'))


if __name__ == '__main__':
    main()
