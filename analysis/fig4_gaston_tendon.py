#!/usr/bin/env python3
"""Fig 4a: GASTON isodepth on the tendon sections, used as the external published-method
comparison for the transferred meninges coordinate.

The tendon's other coordinate (geodesic distance from the tissue surface) is a method
developed for this dataset, so comparing the transferred meninges construction against it
alone is an internal comparison. GASTON (Chitra et al., isodepth from a neural field over
spatial coordinates and expression) is a published external method, run here with the same
settings used for the meninges isodepth (eight restarts), so the comparison is symmetric
across the two tissues.

Settings, matching the meninges run:
  num_dims = 8 top Pearson-residual components   (parse_adata.get_top_pearson_residuals)
  isodepth_arch = [20, 20], expr_arch = [20, 20]
  num_epochs = 8000, checkpoint = 1000, num_restarts = 8, adam, lr = 1e-3
  isodepth read out at nl = 3 and nl = 4 domains (dp_related.get_isodepth_labels)
Each tendon section's spots and counts are read from data_derived/skin_<section>_markerfree.npz,
the same file used by the other Fig 4 panels, giving an identical spot set across panels.

Outputs
  results/fig4_gaston_isodepth.npz    <section>_isodepth_nl3 / _nl4 / _xy / _loss
  results/fig4_gaston_agreement.csv   rank agreement of GASTON isodepth with the transferred
                                      centreline and with the tissue-surface field

The response-surface comparison under GASTON isodepth is run as a separate step so that this
training step can be reused.
"""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
import os, time, json
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import spearmanr
import torch

DD, RES = f"{DER_}", f"{RES_}"
SECTIONS = ["CTRL1", "CTRL2", "INMPP1", "INMPP1-2"]
NUM_DIMS, EPOCHS, CHECKPOINT, RESTARTS = 8, 8000, 1000, 8
ISO_ARCH, EXPR_ARCH = [20, 20], [20, 20]
OUTROOT = f"{DATA_}/work/gaston_tendon"

from gaston import neural_net, parse_adata, process_NN_output, dp_related
import gaston.neural_net as gnn
torch.serialization.add_safe_globals([gnn.GASTON])
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

out, rows = {}, []
t0 = time.time()
for sec in SECTIONS:
    z = np.load(f"{DD}/skin_{sec}_markerfree.npz", allow_pickle=True)
    xy = np.asarray(z["xy_um"], float)
    M = sp.csr_matrix((z["counts_data"], z["counts_indices"], z["counts_indptr"]),
                      shape=tuple(z["counts_shape"]))
    genes = np.array([str(g) for g in z["genes"]])
    counts = np.asarray(M.todense(), dtype=np.float64)
    keep = counts.sum(1) > 0
    assert keep.all(), f"{sec}: {int((~keep).sum())} zero-count spots in the input"
    A = parse_adata.get_top_pearson_residuals(NUM_DIMS, counts, xy, genes, clip=0.01)
    assert np.isfinite(A).all(), f"{sec}: non-finite Pearson residuals"
    S_t, A_t = neural_net.load_rescale_input_data(xy, A)
    sec_dir = f"{OUTROOT}/{sec}"
    os.makedirs(sec_dir, exist_ok=True)
    losses = []
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for seed in range(RESTARTS):
        sd = f"{sec_dir}/rep{seed}"
        os.makedirs(sd, exist_ok=True)
        _, loss_list = neural_net.train(S_t, A_t, S_hidden_list=ISO_ARCH,
                                        A_hidden_list=EXPR_ARCH, epochs=EPOCHS,
                                        checkpoint=CHECKPOINT, device=dev, save_dir=sd,
                                        optim="adam", lr=1e-3, seed=seed, save_final=True)
        losses.append(float(loss_list[-1]))
        print(f"{sec} seed {seed} final_loss={losses[-1]:.4f} "
              f"elapsed={round(time.time()-t0)}s", flush=True)
    model, A_p, S_p = process_NN_output.process_files(sec_dir)
    out[f"{sec}_xy"] = xy
    out[f"{sec}_loss"] = np.array(losses)
    rec = dict(section=sec, n_spots=int(len(xy)), best_loss=float(np.min(losses)),
               worst_loss=float(np.max(losses)))
    for nl in (3, 4):
        iso, lab = dp_related.get_isodepth_labels(model, A_p, S_p, nl)
        iso = np.asarray(iso, float)
        out[f"{sec}_isodepth_nl{nl}"] = iso
        out[f"{sec}_labels_nl{nl}"] = np.asarray(lab).astype(int)
        rec[f"rho_centreline_nl{nl}"] = round(float(abs(spearmanr(
            iso, np.load(f"{RES}/fig4_coord_geometry.npz")[f"{sec}_off"])[0])), 4)
        rec[f"rho_surface_nl{nl}"] = round(float(abs(spearmanr(
            iso, np.asarray(z["R"], float))[0])), 4)
        print(f"{sec} nl={nl}: iso [{iso.min():.3f},{iso.max():.3f}] "
              f"|rho| vs centreline {rec[f'rho_centreline_nl{nl}']:.3f} "
              f"vs surface {rec[f'rho_surface_nl{nl}']:.3f}", flush=True)
    rows.append(rec)

np.savez_compressed(f"{RES}/fig4_gaston_isodepth.npz", **out)
pd.DataFrame(rows).to_csv(f"{RES}/fig4_gaston_agreement.csv", index=False)
print("WROTE fig4_gaston_isodepth.npz / fig4_gaston_agreement.csv "
      f"elapsed={round(time.time()-t0)}s")

# Adds a column to fig4_gaston_agreement.csv: agreement (|Spearman rho|) of the nl=3 isodepth
# with the along-backbone coordinate computed by fig4_along_coord.py.
from scipy.stats import spearmanr as _sp
_al = open(f"{HERE_}/fig4_along_coord.py").read()
_ns = {}
exec(compile("import numpy as np\nfrom sklearn.cluster import KMeans\nfrom scipy.sparse import csr_matrix\n"
             "from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra\nfrom scipy.spatial import cKDTree\n"
             + _al[_al.index("def along_backbone_coord"):], "along", "exec"), _ns)
_ag = pd.read_csv(f"{RES}/fig4_gaston_agreement.csv")
_ag["rho_along_gaston"] = [round(float(abs(_sp(out[f"{s}_isodepth_nl3"],
                           _ns["along_backbone_coord"](out[f"{s}_xy"][:, 0], out[f"{s}_xy"][:, 1]))[0])), 4)
                           for s in _ag.section]
_ag.to_csv(f"{RES}/fig4_gaston_agreement.csv", index=False)
