#!/usr/bin/env python3
"""Run every analysis script in dependency order.

    STPI_DATA=<data folder> python analysis/run_all.py [-j N] [--from STAGE]

Scripts in the same stage do not depend on each other and run in parallel (N at a
time, default 6, 2 BLAS threads each). Logs go to analysis/logs/. GASTON
(fig4_gaston_tendon.py) needs the gaston package; set STPI_GASTON_PYTHON to the
interpreter of an environment that has it, otherwise this script is skipped and the
isodepth table in results/ is used as is.
"""
import os, sys, subprocess, time
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = [
    # inputs built from the raw data
    ["skin_markerfree_coord", "fivestage_normR", "gw6_two_sections"],
    # scans that read only the raw data
    ["multiGW_scan", "merfish_strict_grid", "fig5_audit", "fig5_null", "sim_calibration",
     "stage_feasible_counts", "xtech_nmatched_geneintersect", "fig3_xtech_genelayer",
     "xtech_genelayer_gw6", "matched_depth_bin_bias", "fivestage_random_vs_checker",
     "depth_matched_sscan_5stage", "fig1d_tangential_null", "fig3_coord_arbitration",
     "stage_matched_n", "K_sensitivity", "multiGW_bootstrap",
     "multiGW_bootstrap_block400", "best_size_bootstrap", "block_size_extended",
     "fig3_merfish_gated", "fig3_robustness_scan", "fig4_coord_maps", "fig4_coord_construction"],
    # tables built from the tables above
    ["depth_ceiling", "xtech_genelayer_all", "figS5_support", "fig4_gaston_tendon"],
    ["fig5_buffer3", "fig4_tendon_null", "fig4_tendon_null_surface", "fig4_coord_transfer", "fig4_tendon_scan"],
    ["fig5_pred_maps", "decoded_maps"],
]

def run(name):
    py = sys.executable
    if name == "fig4_gaston_tendon":
        py = os.environ.get("STPI_GASTON_PYTHON")
        if not py:
            return name, "skipped (STPI_GASTON_PYTHON not set)", 0.0
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    os.makedirs(f"{HERE}/logs", exist_ok=True)
    t0 = time.time()
    with open(f"{HERE}/logs/{name}.log", "w") as lg:
        rc = subprocess.call([py, f"{HERE}/{name}.py"], cwd=HERE, env=env, stdout=lg, stderr=subprocess.STDOUT)
    return name, "ok" if rc == 0 else f"failed (exit {rc}, see logs/{name}.log)", (time.time() - t0) / 60

if __name__ == "__main__":
    n = int(sys.argv[sys.argv.index("-j") + 1]) if "-j" in sys.argv else 6
    first = int(sys.argv[sys.argv.index("--from") + 1]) if "--from" in sys.argv else 1
    for k, stage in enumerate(STAGES, 1):
        if k < first:
            continue
        with ThreadPoolExecutor(n) as ex:
            for name, status, minutes in ex.map(run, stage):
                print(f"stage {k}  {name:34s} {status:12s} {minutes:5.1f} min", flush=True)
                if status.startswith("failed"):
                    sys.exit(1)
