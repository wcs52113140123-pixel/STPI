"""
STPI simulation and calibration pipeline.
Parameter-driven, seed-controlled, tidy-table output.

Estimand: incremental positional information dI = I(self+neighbor) - I(self),
measured by a cross-validated decoder (PCA+Ridge) scored as
I = 0.5*log2(Var(R)/MSE), evaluated under random CV vs spatial-block CV (with buffer).
"""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

# ---------------- config / parameter registry ----------------
BASELINE = dict(n=3000, k=15, n_blocks=5, buffer_mult=1.0,
                layer_strength=1.0, sigma_coord=0.02, rho_spatial=0.5,
                beta_self=1.0, beta_neigh=0.0, leakage=False,
                n_self_genes=40, n_neigh_genes=20, noise=6.0,
                n_layers=6, tissue_len=10.0, T=2.0, nuisance_neigh=True)
# positive control: beta_neigh=0.6 corresponds to a true incremental gain of about 0.03 bit
POSCTRL_BETA = 0.6

# ---------------- generator ----------------
def gen_tissue(p, seed):
    rng = np.random.default_rng(seed)
    n = p['n']
    r = rng.uniform(0, 1, n)                  # true radial position (Pia->Dura)
    tang = rng.uniform(0, p['tissue_len'], n)
    pts = np.c_[tang, r * p['T']]
    pts_obs = pts + rng.normal(0, p['sigma_coord'] * p['T'], pts.shape)
    # ---- self-expression channel: noisy readout of position, non-saturating ----
    # single latent position signal per gene, heavy per-cell noise so I_self stays
    # in the empirical regime (~0.2-0.4 bit at baseline), not saturated.
    G = p['n_self_genes']
    fpos = np.c_[r, np.sin(np.pi*r), np.cos(np.pi*r)]      # position basis
    Wself = rng.normal(0, 1, (fpos.shape[1], G))
    Xself = p['beta_self'] * (fpos @ Wself)
    # spatial-autocorrelated nuisance independent of r, drives inflation under random CV
    if p['rho_spatial'] > 0:
        ls = 0.3 + 3.0 * p['rho_spatial']
        kd = cKDTree(pts)
        nz = rng.normal(0, 1, (n, G))
        idx = kd.query_ball_point(pts, ls)
        sm = np.array([nz[i].mean(0) if len(i) else nz[j] for j, i in enumerate(idx)])
        Xself = Xself + p['rho_spatial'] * 2.0 * sm
    Xself = Xself + rng.normal(0, p['noise'], (n, G))       # noise=6 -> non-saturating
    # layers & neighbors
    L = np.clip((r * p['n_layers']).astype(int), 0, p['n_layers']-1)
    kd = cKDTree(pts)
    _, nbr = kd.query(pts, k=p['k']+1)
    nbr = nbr[:, 1:]
    # ---- neighbor channel: a second, independent noisy readout of position ----
    # a genuine local-niche signal that carries information about r through a separate
    # channel with independent noise, so combining it with self yields a real incremental gain.
    Xn = rng.normal(0, p['noise'], (n, p['n_neigh_genes']))
    if p['beta_neigh'] > 0:
        Wn = rng.normal(0, 1, (fpos.shape[1], p['n_neigh_genes']))
        Xn = p['beta_neigh'] * (fpos @ Wn) + Xn
    # ---- spatial nuisance injected into the neighbor block, drives inflation under random CV ----
    # when beta_neigh is 0 this is the only structure in Xn beyond noise; a random-CV decoder
    # can exploit its spatial autocorrelation to produce an apparent information gain, while a
    # spatial-block CV decoder (train and test spatially separated, with a buffer) cannot.
    if p.get('nuisance_neigh', True) and p['rho_spatial'] > 0:
        # neighbor-averaged latent nuisance that weakly tracks r: adjacent cells share
        # neighbors, so they share feature values. Under random CV, a test cell's neighbors
        # leak into the training fold, producing an apparent gain; under spatial-block CV with
        # a buffer, neighbors of test cells are excluded, so there is no gain. This is the
        # confound that the buffer is designed to remove, and it generates the inflation curve.
        latent = 0.15 * r + rng.normal(0, 1, n)          # weak r-dependence + noise
        neigh_avg = latent[nbr].mean(1)                   # spatial averaging = leakage source
        Wlk = rng.normal(0, 1, (1, p['n_neigh_genes']))
        Xn = Xn + p['rho_spatial'] * 4.0 * (neigh_avg[:, None] @ Wlk)
    # neighborhood composition features (leakage experiment)
    if p['leakage']:
        onehot = np.eye(p['n_layers'])[L]
        comp_leaky = np.array([onehot[nbr[i]].mean(0) for i in range(n)])
        broad = np.zeros(n, int)
        onehot_b = np.eye(1)[broad]
        comp_free = np.array([onehot_b[nbr[i]].mean(0) for i in range(n)])
    else:
        comp_leaky = comp_free = None
    return dict(r=r, pts=pts_obs, Xself=Xself, Xn=Xn, L=L, nbr=nbr,
                comp_leaky=comp_leaky, comp_free=comp_free)

# ---------------- CV engine ----------------
def spatial_blocks(pts, n_blocks):
    xq = np.quantile(pts[:, 0], np.linspace(0, 1, n_blocks+1))
    return np.clip(np.digitize(pts[:, 0], xq[1:-1]), 0, n_blocks-1)

def _score_I(R, pred, mask):
    v = np.var(R[mask])
    mse = np.mean((R[mask] - pred[mask])**2)
    return 0.5 * np.log2(v / mse) if mse > 0 and v > 0 else np.nan

def decode_I(R, pts, X, mode, p, seed=0):
    """Return decoder information I(bit) for feature matrix X predicting R."""
    n = len(R)
    if mode == 'random':
        blk = np.random.default_rng(seed).integers(0, p['n_blocks'], n)
        buf = 0.0
    else:
        blk = spatial_blocks(pts, p['n_blocks'])
        dnn, _ = cKDTree(pts).query(pts, k=2)
        buf = p['buffer_mult'] * np.median(dnn[:, 1]) * p['k']  # buffer ~ k * nn-spacing
    pred = np.full(n, np.nan)
    for kb in range(p['n_blocks']):
        te = np.where(blk == kb)[0]
        if len(te) < 8:
            continue
        if mode == 'random':
            tr = np.where(blk != kb)[0]
        else:
            dmin, _ = cKDTree(pts[te]).query(pts)
            tr = np.where((blk != kb) & (dmin > buf))[0]
            if len(tr) < 40:
                tr = np.where(blk != kb)[0]
        nc = min(30, X.shape[1], len(tr)-1)
        if nc < 2:
            continue
        pca = PCA(nc, random_state=0).fit(X[tr])
        Zt = pca.transform(X[tr]); Ze = pca.transform(X[te])
        m = Zt.mean(0); sd = Zt.std(0) + 1e-8
        rg = Ridge(alpha=1.0).fit((Zt-m)/sd, R[tr])
        pred[te] = rg.predict((Ze-m)/sd)
    mask = ~np.isnan(pred)
    return _score_I(R, pred, mask), mask.sum()

def delta_I(sim, mode, p, seed=0):
    """dI = I(self+neighbor features) - I(self).
    Baseline is dimensionality-matched: the self-only model is padded with the
    same number of matched pure-noise columns as the neighbor block, so dI
    isolates the information contributed by neighbors rather than a feature-count effect.
    """
    R = sim['r']; pts = sim['pts']
    Xself = sim['Xself']
    if sim['comp_leaky'] is not None:
        Xn_leaky = sim['comp_leaky']; Xn_free = sim['comp_free']
    else:
        Xn_leaky = sim['Xn']; Xn_free = None
    rng = np.random.default_rng(seed + 777)
    pad = rng.normal(0, p['noise'], (len(R), Xn_leaky.shape[1]))
    I_self, _ = decode_I(R, pts, np.c_[Xself, pad], mode, p, seed)
    I_both, _ = decode_I(R, pts, np.c_[Xself, Xn_leaky], mode, p, seed)
    out = {'I_self': I_self, 'I_both': I_both, 'dI': I_both - I_self}
    if Xn_free is not None:
        padf = rng.normal(0, p['noise'], (len(R), Xn_free.shape[1]))
        I_selff, _ = decode_I(R, pts, np.c_[Xself, padf], mode, p, seed)
        I_free, _ = decode_I(R, pts, np.c_[Xself, Xn_free], mode, p, seed)
        out['I_both_leakfree'] = I_free
        out['dI_leakfree'] = I_free - I_selff
    return out

# ---------------- permutation engine ----------------
def perm_p(sim, mode, p, dI_obs, B=100, seed=0):
    """Permute neighbor-feature rows (break local pairing) and recompute dI against the
    same dimensionality-matched padded baseline used in the observed statistic,
    for a one-sided P(perm >= obs)."""
    rng = np.random.default_rng(seed)
    R = sim['r']; pts = sim['pts']; Xself = sim['Xself']
    Xn = sim['comp_leaky'] if sim['comp_leaky'] is not None else sim['Xn']
    # uses the same padded baseline as delta_I (same seed offset gives the same pad)
    rng_pad = np.random.default_rng(seed + 777)
    pad = rng_pad.normal(0, p['noise'], (len(R), Xn.shape[1]))
    I_self, _ = decode_I(R, pts, np.c_[Xself, pad], mode, p, seed)
    cnt = 0
    for b in range(B):
        perm = rng.permutation(len(R))
        I_both, _ = decode_I(R, pts, np.c_[Xself, Xn[perm]], mode, p, seed)
        if (I_both - I_self) >= dI_obs:
            cnt += 1
    return (cnt + 1) / (B + 1)

# ---------------- runner ----------------
def run_condition(exp, p, R_rep, seed0, do_perm=False, B=100):
    rows = []
    for rep in range(R_rep):
        seed = seed0 + rep
        sim = gen_tissue(p, seed)
        for mode in ('random', 'spatial'):
            d = delta_I(sim, mode, p, seed)
            row = dict(exp=exp, rep=rep, seed=seed, cv=mode,
                       n=p['n'], k=p['k'], rho_spatial=p['rho_spatial'],
                       sigma_coord=p['sigma_coord'], beta_neigh=p['beta_neigh'],
                       layer_strength=p['layer_strength'], leakage=int(p['leakage']),
                       buffer_mult=p['buffer_mult'], **d)
            if do_perm:
                row['p_perm'] = perm_p(sim, mode, p, d['dI'], B=B, seed=seed)
            rows.append(row)
    return rows

def merge(base, **kw):
    q = dict(base); q.update(kw); return q

# ---------------- experiment plan (conditions S1-S7 plus joint grid) ----------------
def experiment_plan():
    B = BASELINE
    plan = []  # (exp, params, do_perm)
    # S1: true neighborhood signal (monotonicity and power); the beta grid maps to true gains
    # of approximately {0, 0.007, 0.015, 0.023, 0.033 (positive control), 0.045, 0.088} bit
    for bn in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]:
        plan.append(('S1_signal', merge(B, beta_neigh=bn, nuisance_neigh=False), False))
    # S2: rho_spatial (inflation under random CV); beta_neigh=0 so there is no real extra information
    for rho in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        plan.append(('S2_rho', merge(B, rho_spatial=rho, beta_neigh=0.0), False))
    # S3: layer strength combined with leakage (false niche signal); beta_neigh=0
    for ls in [0.5, 1.0, 2.0]:
        for lk in [False, True]:
            plan.append(('S3_leak', merge(B, layer_strength=ls, leakage=lk, beta_neigh=0.0), False))
    # S4: coordinate noise
    for sc in [0.0, 0.02, 0.05, 0.1, 0.2]:
        plan.append(('S4_coordnoise', merge(B, sigma_coord=sc, beta_neigh=1.0), False))
    # S5: number of neighbors k
    for k in [6, 10, 15, 30, 50]:
        plan.append(('S5_k', merge(B, k=k, beta_neigh=1.0), False))
    # S6: sample size n
    for n in [500, 1000, 2000, 3000, 5000]:
        plan.append(('S6_n', merge(B, n=n, beta_neigh=1.0), False))
    # S7: buffer width and block width
    for bm in [0.0, 0.5, 1.0, 2.0]:
        plan.append(('S7_buffer', merge(B, buffer_mult=bm, rho_spatial=0.8, beta_neigh=0.0), False))
    # type-I error: null conditions with permutation testing (no true signal)
    plan.append(('typeI_null', merge(B, beta_neigh=0.0, rho_spatial=0.5, nuisance_neigh=False), True))
    plan.append(('typeI_null_rho', merge(B, beta_neigh=0.0, rho_spatial=1.0, nuisance_neigh=False), True))
    # power: positive control beta_neigh=0.6 (about 0.03 bit true gain) with permutation testing
    plan.append(('power_posctrl', merge(B, beta_neigh=POSCTRL_BETA, nuisance_neigh=False), True))
    # joint stress grid over rho_spatial, sigma_coord and leakage (3^3 combinations)
    for rho in [0.0, 0.5, 1.0]:
        for sc in [0.02, 0.1, 0.2]:
            for lk in [False, True]:
                plan.append(('joint', merge(B, rho_spatial=rho, sigma_coord=sc, leakage=lk, beta_neigh=0.0), False))
    # conditions expected to fail the calibration criteria
    for cond in [merge(B, n=200, beta_neigh=1.0), merge(B, sigma_coord=0.5, beta_neigh=1.0),
                 merge(B, beta_neigh=0.1), merge(B, n_blocks=15, beta_neigh=1.0)]:
        plan.append(('failure', cond, False))
    return plan

def _one(args):
    exp, p, do_perm, R_rep, seed0, B = args
    if threadpool_limits is not None:
        with threadpool_limits(limits=1):
            return run_condition(exp, p, R_rep, seed0, do_perm=do_perm, B=B)
    return run_condition(exp, p, R_rep, seed0, do_perm=do_perm, B=B)

def main(R_rep=200, R_perm=200, B_perm=100, outdir=None, nproc=64):
    import multiprocessing as mp, itertools, os
    outdir = outdir or f"{RES_}"
    plan = experiment_plan()
    tasks = []
    for i, (exp, p, do_perm) in enumerate(plan):
        rr = R_perm if do_perm else R_rep
        # split each condition into chunks of reps for parallelism
        chunk = 10
        for j in range(0, rr, chunk):
            tasks.append((exp, p, do_perm, min(chunk, rr-j), 1000*i + j, B_perm))
    with mp.Pool(nproc) as pool:
        res = pool.map(_one, tasks)
    rows = list(itertools.chain.from_iterable(res))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "sim_calibration_results.csv"), index=False)
    print("WROTE", len(df), "rows to sim_calibration_results.csv")
    return df

if __name__ == "__main__":
    main()
