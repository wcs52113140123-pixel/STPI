#!/usr/bin/env python3
"""skin_markerfree_coord.py: marker-free geometric coordinate for skin Visium.
R is the geodesic depth from the tissue surface (boundary-distance Laplacian/Dijkstra field).
Construction uses ONLY spot coordinates and tissue-edge detection (angular-gap boundary).
NO transcriptome input enters R. Keratin and collagen markers are used only afterwards to
validate that the surface (R~0) is keratinocyte-high (epidermis) and the interior (R~1) is
collagen-high (dermis): a direction check, not a construction input.
Anchored to Visium native pitch = 100um. HVG=5000. Outputs skin_<section>_markerfree.npz.
"""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
import numpy as np, scipy.sparse as sp, scipy.io, pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csgraph
from scipy.stats import spearmanr
base=f"{DATA_}/skin_GSE245179"
dd=DER_
SAMPLES=["GSM7836297_CTRL1","GSM7836298_CTRL2","GSM7836299_INMPP1","GSM7836300_INMPP1-2"]
def boundary_by_angular_gap(P,k=12):
    tree=cKDTree(P);dnn,idx=tree.query(P,k=k+1);med=np.median(dnn[:,1]);bnd=np.zeros(len(P),bool)
    for i in range(len(P)):
        nb=idx[i,1:];nb=nb[dnn[i,1:]<3*med]
        if len(nb)<3: bnd[i]=True;continue
        ang=np.sort(np.arctan2(P[nb,1]-P[i,1],P[nb,0]-P[i,0]))
        if np.diff(np.concatenate([ang,[ang[0]+2*np.pi]])).max()>np.pi*0.9: bnd[i]=True
    return bnd
for s in SAMPLES:
    mtx=scipy.io.mmread(f"{base}/{s}_matrix.mtx.gz").tocsr()
    pos=pd.read_csv(f"{base}/{s}_tissue_positions_list.csv.gz",header=None,
        names=["barcode","in_tissue","row","col","y_px","x_px"])
    bc=pd.read_csv(f"{base}/{s}_barcodes.tsv.gz",header=None)[0].values
    genes=pd.read_csv(f"{base}/{s}_features.tsv.gz",sep="\t",header=None)[1].values
    pos=pos.set_index("barcode").reindex(bc);sel=pos.in_tissue.values==1
    P=pos[["x_px","y_px"]].values.astype(float)[sel]
    tree=cKDTree(P);dnn,_=tree.query(P,k=2);med=np.median(dnn[:,1]);umpp=100.0/med
    pairs=tree.query_pairs(r=1.5*med,output_type='ndarray');n=len(P)
    W=sp.coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(n,n));W=(W+W.T).tocsr()
    bnd=boundary_by_angular_gap(P)
    dist=csgraph.dijkstra(W,indices=np.where(bnd)[0],min_only=True)
    fin=np.isfinite(dist);dist[~fin]=dist[fin].max()
    R=(dist-dist.min())/(dist.max()-dist.min())
    def score(marks):
        ix=[np.where(genes==mm)[0][0] for mm in marks if (genes==mm).any()]
        sub=mtx[ix][:,sel].toarray().astype(float);lib=np.asarray(mtx[:,sel].sum(0)).ravel();lib[lib==0]=1
        return np.log1p(sub/lib*np.median(lib)).mean(0)
    rk=spearmanr(R,score(["Krt14","Krt5","Krt10","Krt1","Krt15"]))[0]
    rc=spearmanr(R,score(["Col1a1","Col1a2","Col3a1","Dcn","Lum"]))[0]
    X=mtx[:,sel].tocsr().astype(float);lib=np.asarray(X.sum(0)).ravel();lib[lib==0]=1
    Xn=X.multiply(1.0/lib).multiply(np.median(lib)).tocsr();Xn.data=np.log1p(Xn.data)
    gvar=np.asarray(Xn.power(2).mean(1)).ravel()-np.asarray(Xn.mean(1)).ravel()**2
    hvg=np.argsort(gvar)[::-1][:5000];Xh=X[hvg].T.tocsr()
    np.savez_compressed(f"{dd}/skin_{s.split('_')[1]}_markerfree.npz",
        counts_data=Xh.data,counts_indices=Xh.indices,counts_indptr=Xh.indptr,counts_shape=Xh.shape,
        genes=genes[hvg],xy_um=P*umpp,R=R,umpp=umpp,native_pitch_um=100.0)
    print(f"{s.split('_')[1]}: {Xh.shape[0]} spots, R.Krt={rk:+.3f} R.Col={rc:+.3f}")
