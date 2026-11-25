# analysis/

Scripts that write every table in `results/` and the two geometry files in `data_derived/`,
from the raw data. Run them all, in dependency order, with

```
STPI_DATA=<data folder> python analysis/run_all.py -j 12
```

About 50 minutes on 12 cores. Each script can also be run on its own from this folder.

## Inputs

`STPI_DATA` must contain:

| Folder | Content | Source |
|---|---|---|
| `gw6_annotated/`, `gaston_gw16/`, `gaston_multi/` | human fetal meninges, Stereo-seq and MERFISH | Li et al. 2026 (*Cell*); human genetic resource data, available from the authors on request, not redistributed |
| `skin_GSE245179/` | injured mouse tendon, Visium | GEO GSE245179 |
| `kidney_GSE273978/` | mouse kidney | GEO GSE273978 |
| `intestine_GSE245274/` | mouse intestine | GEO GSE245274 |
| `cerebellum_GSE296375/` | mouse cerebellum | GEO GSE296375 |
| `retina_GSE248517/` | mouse retina | GEO GSE248517 |
| `retina_GSE309408_finch/` | finch retina | GEO GSE309408 |
| `derived/` | MERFISH cell metadata and the shared gene list; `skin_<section>_markerfree.npz` written by stage 1 | built from the folders above |

Other variables: `STPI_RESULTS` (default `../results`), `STPI_DERIVED` (default `$STPI_DATA/derived`),
`STPI_DATA_DERIVED` (default `../data_derived`), and `STPI_GASTON_PYTHON`, the interpreter of an
environment with the `gaston` package; without it the GASTON comparison is skipped and the stored
isodepth table is used.

## Reproducibility

Every random step is seeded: PCA with `random_state=0`, per-section streams from SHA-256 of the section
name (`stpi.seed_of`), and the Moran eigenvector basis is computed on one thread so that it does not
change with the number of BLAS threads. Two runs give identical tables.

## Scripts

| Stage | Script | Writes | Used in |
|---|---|---|---|
| 1 | `skin_markerfree_coord.py` | `skin_<section>_markerfree.npz (derived input)` | inputs of Fig 4, S4 |
| 1 | `fivestage_normR.py` | `fivestage_normR.npz` | Fig 1 B, S1 A |
| 1 | `gw6_two_sections.py` | `gw6_two_sections.npz` | Fig 1 B, S1 A |
| 2 | `multiGW_scan.py` | `multiGW_T.json`, `multiGW_grid_all.csv` | Fig 1 E, Fig 2 A, B, D, S2 A, D |
| 2 | `merfish_strict_grid.py` | `merfish_grid_all.csv` | Fig 3 C, S3 C |
| 2 | `fig5_audit.py` | `fieldwide_audit.csv`, `fig5_buffered_cv_all.csv` | Fig 5 A, C, D, E, S5 C, D |
| 2 | `fig5_null.py` | `fig5_fieldwide_null_msr.csv`, `fig5_fieldwide_null_msr.npz` | Fig S5 A |
| 2 | `sim_calibration.py` | `sim_calibration_results.csv` | Fig S1 C, D, E |
| 2 | `stage_feasible_counts.py` | `stage_feasible_counts.csv` (intermediate) | Fig S2 D |
| 2 | `xtech_nmatched_geneintersect.py` | `xtech_nmatched_geneintersect.csv` | Fig 3 D, S3 D |
| 2 | `fig3_xtech_genelayer.py` | `fig3_xtech_genelayer.csv` (intermediate) | Fig S3 D |
| 2 | `xtech_genelayer_gw6.py` | `fig3_xtech_genelayer_GW6.csv` (intermediate) | Fig S3 D |
| 2 | `matched_depth_bin_bias.py` | `matched_depth_bin_bias.csv` | Fig S2 |
| 2 | `fivestage_random_vs_checker.py` | `fivestage_random_vs_checker_C.csv` | Fig 1 C |
| 2 | `depth_matched_sscan_5stage.py` | `depth_matched_sscan_5stage.csv` | Fig S2 A, D |
| 2 | `fig1d_tangential_null.py` | `tangential_null_draws.npz` | Fig 1 D |
| 2 | `fig3_coord_arbitration.py` | `fig3_coord_arbitration.csv`, `fig3_coord_grids.csv` | Fig 3 A, S3 A |
| 2 | `stage_matched_n.py` | `stage_matched_n_reps.csv`, `stage_matched_n_summary.csv` (intermediate) | Fig 2 F |
| 2 | `K_sensitivity.py` | `K_sensitivity.csv` | Fig 2 E, S2 E, F |
| 2 | `multiGW_bootstrap.py` | `multiGW_bootstrap.csv`, `multiGW_effects_by_stage.csv` | Fig S2 C |
| 2 | `multiGW_bootstrap_block400.py` | `multiGW_effects_by_stage_block400.csv`, `multiGW_bootstrap_block400.csv` (intermediate) | Fig 2 C |
| 2 | `best_size_bootstrap.py` | `best_size_boot_raw.json`, `multiGW_T_check.json` (intermediate), `best_size_rawI_bootstrap.csv` (intermediate), `best_size_spearman.json` (intermediate), `best_size_summary.csv` (intermediate) | Fig S2 B |
| 2 | `block_size_extended.py` | `block_size_extended_summary.csv`, `block_size_extended.csv` (intermediate) | Fig S2 C |
| 2 | `fig3_merfish_gated.py` | `merfish_nmatched_gated.csv` | Fig 3 D |
| 2 | `fig3_robustness_scan.py` | `fig3_bufferedcv_5stage.csv`, `fig3_altcoord_5stage.csv` (intermediate) | Fig 3 B, S3 B |
| 2 | `fig4_coord_maps.py` | `fig4_coord_maps.npz` | Fig 4 A |
| 2 | `fig4_coord_construction.py` | `fig4_coord_geometry.npz`, `fig4_coord_validation.csv` (intermediate) | Fig 4 A, S4 A |
| 3 | `depth_ceiling.py` | `depth_ceiling_triple_collinearity.csv` | Fig S2 D |
| 3 | `xtech_genelayer_all.py` | `xtech_genelayer_all.csv` | Fig S3 D |
| 3 | `figS5_support.py` | `fig5_blocksize_sensitivity.csv`, `fig5_deletion_control.csv`, `fig5_pred_maps_all.npz`, `fig5_section_covariates.csv` | Fig S5 B, C, D, E |
| 3 | `fig4_gaston_tendon.py` | `fig4_gaston_agreement.csv`, `fig4_gaston_isodepth.npz` | Fig S4 C, E |
| 4 | `fig5_buffer3.py` | `fig5_buffered_cv_all.csv`, `fig5_deletion_control.csv`, `fig5_buffer3.csv` (intermediate) | Fig 5 E, S5 E |
| 4 | `fig4_tendon_null.py` | `fig4_tendon_null.csv`, `fig4_tendon_null.npz` | Fig S4 D |
| 4 | `fig4_tendon_null_surface.py` | `fig4_tendon_null_surface.csv`, `fig4_tendon_null_surface.npz` | Fig S4 D |
| 4 | `fig4_coord_transfer.py` | `fig4_coord_transfer.csv`, `fig4_coord_agreement.csv` (intermediate) | Fig 4 D, S4 B |
| 4 | `fig4_tendon_scan.py` | `fig4_tendon_cv.csv`, `fig4_tendon_cv_along.csv`, `fig4_tendon_cv_centreline.csv`, `fig4_tendon_cv_gaston.csv` | Fig 4 B, C, E, S4 C, E |
| 5 | `fig5_pred_maps.py` | `fig5_pred_maps.npz` | Fig 5 B |
| 5 | `decoded_maps.py` | `decoded_maps_meninges.npz`, `decoded_maps_tendon.npz`, `decoded_maps_check.csv` (intermediate) | Fig S1 B, Fig S4 B |

Helper modules: `_paths.py` (folders), and `fig5_buffered_all.py`, `fig3_altcoord_scan.py`,
`fig4_along_coord.py`, `multiGW_bootstrap.py`, `xtech_nmatched.py`, whose functions other scripts load.
`sections_fig5.csv` lists the 18 published sections of Fig 5, their folder and reader.

`stereo_hvg5000.R` (R, Seurat) shows how the GW8 to GW16 Stereo-seq matrices in `gaston_multi/` and
`gaston_gw16/` were restricted to 5,000 genes: the 5,000 most variable genes of each section, selected
with `FindVariableFeatures(selection.method = "vst")`. Those matrices are inputs, so `run_all.py` does
not call it. Run it with `Rscript stereo_hvg5000.R <STPI_DATA> <output folder>`; with Seurat 5.3.0 it
returns the stored gene sets at GW10, GW12 and GW16, and 4,998 of 5,000 genes at GW8.
