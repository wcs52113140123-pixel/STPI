# Figure code

One script per figure. Each script reads only `results/` and `data_derived/` in this
repository and writes to `figures/out/`. The figures themselves are not part of the
repository; running the scripts regenerates them.

```
cd figures
python fig1.py      # and fig2.py ... fig5.py, figS1.py ... figS5.py
python keys.py      # colour and mark keys placed beside panels
```

Tested with Python 3.13, matplotlib 3.11, numpy 2.5, pandas 2.3, scipy 1.18,
seaborn 0.13 and Pillow 12.3.

Environment variables (all optional): `STPI_RESULTS`, `STPI_DERIVED` (input folders) and
`STPI_FIGOUT` (output folder).

## Fonts

The published figures use Arial. Arial is licensed and is not distributed with the code.
Put `ARIAL*.TTF` in `assets/fonts/` for a byte-identical rebuild; otherwise the scripts fall
back to Liberation Sans, which has the same metrics, so the layout does not change.

## What each script writes

For every panel there are two outputs.

| Output | Content |
|---|---|
| `out/<Fig>/marks/<Fig>_<panel>.png` | the graphic as placed in the figure: data marks only, no text. Titles, axis names, tick labels and keys are set in the figure layout. |
| `out/<Fig>/<Fig>_<panel>.png/.pdf` | the same panel with its title, axes and labels, for checking. |

Colour bars and keys that sit beside a panel are separate images
(`*_colourbar.png`, `*_key*.png`, and `out/keys/`).

| Script | Panels | Notes |
|---|---|---|
| `fig1.py` | B, C, D, E | A is a schematic drawn in the figure. B: five stages scaled by tissue diagonal; the stage labels, leader lines and scale bars are set in the figure. |
| `figS1.py` | A to E | A: GW6 outlines and the short normal lines drawn over them are added in the figure. B: decoded fields at native resolution. |
| `fig2.py` | A to F | |
| `figS2.py` | A to F | the sheet is composed once and each panel is cut from it; E is written as four images, one per pseudospot size. |
| `fig3.py` | A to D | |
| `figS3.py` | A to D | |
| `fig4.py` | A to E | A shows the STPI coordinate row. |
| `figS4.py` | A to E | A is written as one image per section plus the three key marks. |
| `fig5.py` | A to E | |
| `figS5.py` | A to E | |
| `keys.py` | keys | tissue colours, Fig 5 E marks, Fig 1 C swatches. |

Shared code: `_common.py` (paths, fonts, palettes, output helpers) and `_fields.py`
(decoded-field rendering).

## Inputs of each script

Every input is in `results/` or `data_derived/` and is written by a script in `analysis/`
(see `analysis/README.md`).

| Script | Reads |
|---|---|
| `fig1.py` | `fivestage_normR.npz`, `fivestage_random_vs_checker_C.csv`, `gw6_two_sections.npz`, `multiGW_grid_all.csv`, `tangential_null_draws.npz` |
| `figS1.py` | `decoded_maps_meninges.npz`, `fivestage_normR.npz`, `gw6_two_sections.npz`, `sim_calibration_results.csv` |
| `fig2.py` | `stage_matched_n_reps.csv`, `depth_matched_sscan_5stage.csv`, `multiGW_T.json`, `multiGW_effects_by_stage_block400.csv`, `multiGW_grid_all.csv` |
| `figS2.py` | `K_sensitivity.csv`, `block_size_extended_summary.csv`, `stage_matched_n_reps.csv`, `depth_ceiling_triple_collinearity.csv`, `depth_matched_sscan_5stage.csv`, `matched_depth_bin_bias.csv`, `multiGW_bootstrap.csv`, `multiGW_effects_by_stage.csv`, `multiGW_grid_all.csv`, `best_size_boot_raw.json` |
| `fig3.py` | `fig3_bufferedcv_5stage.csv`, `fig3_coord_arbitration.csv`, `merfish_grid_all.csv`, `merfish_nmatched_gated.csv`, `multiGW_grid_all.csv`, `xtech_nmatched_geneintersect.csv` |
| `figS3.py` | `fig3_bufferedcv_5stage.csv`, `fig3_coord_grids.csv`, `xtech_genelayer_all.csv`, `merfish_grid_all.csv`, `multiGW_grid_all.csv` |
| `fig4.py` | `fig4_coord_geometry.npz`, `fig4_coord_maps.npz`, `fig4_coord_transfer.csv`, `fig4_gaston_agreement.csv`, `fig4_gaston_isodepth.npz`, `fig4_tendon_cv_centreline.csv` |
| `figS4.py` | `decoded_maps_tendon.npz`, `fig4_coord_geometry.npz`, `fig4_coord_transfer.csv`, `fig4_tendon_cv.csv`, `fig4_tendon_cv_along.csv`, `fig4_tendon_cv_gaston.csv`, `fig4_tendon_null.csv`, `fig4_tendon_null.npz`, `fig4_tendon_null_surface.csv`, `fig4_tendon_null_surface.npz` |
| `fig5.py` | `fieldwide_audit.csv`, `fig5_buffered_cv_all.csv`, `fig5_pred_maps.npz` |
| `figS5.py` | `fieldwide_audit.csv`, `fig5_blocksize_sensitivity.csv`, `fig5_buffered_cv_all.csv`, `fig5_deletion_control.csv`, `fig5_fieldwide_null_msr.csv`, `fig5_pred_maps_all.npz`, `fig5_section_covariates.csv` |
