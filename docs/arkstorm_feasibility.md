# ARkStorm 2.0 forcing — feasibility memo (2026-08-18)

**Verdict: GO.** The scenario meteorology is public, in the right variables and
resolution, and the gridded-WRF-to-basin pipeline has precedent (USACE HEC-HMS
tutorial; Yuba/DWR/CW3E FIRO work).

## Archive

Huang & Swain (2022) archive their WRF output on NSF DesignSafe, project
**PRJ-3499**, DOI [10.17603/ds2-mzgn-cy51](https://doi.org/10.17603/ds2-mzgn-cy51).
License ODC-BY (attribution only, no registration for individual files). The
needed files sit under `/PRJ-3499/Post/`:

| File | Variable | Resolution |
|---|---|---|
| `wrf_pr_hourly_3km_arkevt_30day_{hist,ftr}.nc` | precipitation | hourly, 3 km |
| `wrf_t2_3km_arkevt_30day_{hist,ftr}.nc` | 2-m temperature | 3 km (verify timestep on open) |

Two scenarios: **ARkHist** (mapped to 2002-02-09..2002-03-12) and **ARkFuture**
(RCP8.5, mapped to 2072-01-11..2072-02-11), ~30 days each. Native grid is WRF
Lambert Conformal Conic (grid parameters in `/Domain_WPS`, `/Namelist`);
reprojection to basin polygons is required. Total download ~1–2 GB (USACE's
pre-cleaned pr+T2 package for both scenarios is 2.12 GB). No THREDDS/OPeNDAP;
flat HTTPS download. `/Analysis Data/` has HUC4-masked precip series (1802
Sacramento, 1804 San Joaquin) usable as a regridding QA check.

## Steps

1. Download the 4 Post/ files + README + Namelist/Domain_WPS.
2. Confirm T2 timestep/units and that `pr` is total precipitation.
3. Reproject 3-km grid to the 22 cohort basin polygons (regionmask/xESMF;
   Caravan ships basin shapefiles — separate Zenodo download).
4. Aggregate hourly → daily (sum precip, mean T2); QA any basin nested in
   HUC 1802/1804 against the archive's own masked series.
5. Map into the model's forcing slots and run inference with the trained
   flood-split checkpoint.

## Effort and risk

**1–2 focused days** given the existing basin-averaging machinery; +1 day if
basin polygons must be fetched and wired. Main risk is interpretational, not
access: the model was trained on ERA5-Land/IMERG/CPC-class forcing and raw WRF
carries its own orographic biases, so results must be framed as a scenario
stress test, not a validated forecast.

Precedent: USACE, [HEC-HMS ARkStorm tutorial](https://www.hec.usace.army.mil/confluence/hmsdocs/hmsguides/gridded-boundary-condition-data/using-the-arkstorm-2-0-meteorology-within-hec-hms);
DRI [ARkStorm@SierraFront 2.0](https://www.dri.edu/project/arkstormsierrafront-2-0/).
