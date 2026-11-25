#!/usr/bin/env python
"""Tests whether Fig 3d depends on restricting both platforms to the 265 shared genes.

Fig 3d compares Stereo-seq and MERFISH at a matched pseudospot count on the shared
265-gene panel, where the Stereo-seq readings are negative. Restricting to the shared
panel could itself limit Stereo-seq (few genes, few retained units), in which case the
comparison would measure the restriction rather than the platforms.

This script reuses xtech_nmatched.py (loaders, pseudospot binning, depth thinning, the
geometric coordinate, spatial block CV, the matched-n bootstrap, the cell size of 100 um
and depth 16, B = 300, seed) via exec on its source, changing only the gene mask. Each
platform is run twice: once on the shared panel (reproducing Fig 3d) and once on its own
native gene set.

Output: results/fig3_xtech_genelayer.csv
"""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
import os
for v in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:
    os.environ[v]="1"
import numpy as np, pandas as pd
from scipy.io import mmread
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

BASE=DATA_
EXPR=f"{DATA_}/gw6_annotated/expr"; DD=f"{DER_}"
GM=f"{DATA_}/gaston_multi"; GG=f"{DATA_}/gaston_gw16"
shared=[l.strip() for l in open(f"{DD}/stereo_merfish_shared_genes.txt")]
GENESET="shared"   # flipped to "native" for the control arm
ORDER=["Fibroblast_Pia","Fibroblast_Arachnoid","Fibroblast_Dura"]
# Common feasible condition for both technologies on the restricted shared-gene panel.
# Constraints: (i) Stereo-seq UMI counts on the shared genes are very low (s=50 um, d=64 gives
# 0 Stereo-seq pseudospots at GW8); (ii) MERFISH sections are physically small, so coarse bins
# are sparse (s=200 um gives only 17-35 MERFISH pseudospots at GW8-12). s=100 um, d=16 is the
# cell where both platforms retain adequate n at every stage (MERFISH 48-413, Stereo-seq
# 82-7803). This is the coarsest feasible matched cell on the shared panel.
SCALE=100.0; DEPTH=16; NB=5; B=300; N_MIN=40
# Stereo-seq coordinates are in units of 0.5 um and MERFISH coordinates in um, so the Stereo-seq
# side is binned at SCALE / STEREO_UM coordinate units: both platforms use SCALE-um pseudospots.
STEREO_UM = 0.5
SCALE_STEREO = SCALE / STEREO_UM

def geometry_R(cx,cy,nseg=12):
    pts=np.c_[cx,cy].astype(float); pr=PCA(2).fit_transform(pts); t=pr[:,0]
    n=len(t)
    nseg=max(3,min(nseg,n//5))          # adaptive: >=5 pts/segment, >=3 segments
    minpts=max(3,n//(nseg*3))
    bins=np.quantile(t,np.linspace(0,1,nseg+1)); bc=[];bm=[]
    for i in range(nseg):
        s=(t>=bins[i])&(t<=bins[i+1])
        if s.sum()>=minpts: bc.append(np.median(t[s]));bm.append(np.median(pr[s,1]))
    if len(bc)<2:                        # fallback: PC2 itself as across-band axis
        off=pr[:,1]
    else:
        center=np.interp(t,np.array(bc),np.array(bm)); off=pr[:,1]-center
    return (off-off.min())/(off.max()-off.min()+1e-9)

def load_merfish(stage):
    sk={"GW8":"GW8_MERFISH","GW10":"GW10_MERFISH","GW12":"GW12_MERFISH","GW16":"GW16_MERFISH"}[stage]
    M=mmread(f"{EXPR}/{sk}_counts.mtx").tocsr()
    genes=[l.strip() for l in open(f"{EXPR}/{sk}_genes.txt")]
    cells=np.array([l.strip() for l in open(f"{EXPR}/{sk}_cells.txt")])
    if M.shape[0]==len(genes): M=M.T
    M=M.tocsr(); cmap={c:i for i,c in enumerate(cells)}
    meta=pd.read_csv(f"{DD}/merfish_meta_{stage}.csv"); meta['Cell_ID_2']=meta['Cell_ID_2'].astype(str)
    meta=meta[meta['Cell_ID_2'].isin(cmap)]
    idx=[cmap[c] for c in meta['Cell_ID_2']]
    X=np.asarray(M[idx].todense()); cx=meta['cx'].values; cy=meta['cy'].values; ct=meta['CellType'].values
    gmask=np.isin(genes,shared) if GENESET=='shared' else np.ones(len(genes),bool)
    return X[:,gmask],np.array(genes)[gmask],cx,cy,ct

def load_stereo(stage):
    if stage=="GW16":
        counts=np.load(f"{GG}/counts_mat.npy"); coords=np.load(f"{GG}/coords_mat.npy")
        genes=np.array([l.strip() for l in open(f"{GG}/genes.txt")])
        clusters=np.load(f"{GG}/clusters_kept.npy",allow_pickle=True)
    else:
        counts=np.asarray(mmread(f"{GM}/{stage}/counts.mtx").T.todense(),dtype=np.float64)
        coords=np.loadtxt(f"{GM}/{stage}/coords.tsv")
        genes=np.array([l.strip() for l in open(f"{GM}/{stage}/genes.txt")])
        clusters=np.array([l.strip() for l in open(f"{GM}/{stage}/clusters.txt")])
    fib=np.isin(clusters,ORDER)
    gmask=np.isin(genes,shared) if GENESET=='shared' else np.ones(len(genes),bool)
    return counts[fib][:,gmask],genes[gmask],coords[fib,0],coords[fib,1]

def pseudospots(cx,cy,X,scale,depth):
    bx=np.floor((cx-cx.min())/scale).astype(int); by=np.floor((cy-cy.min())/scale).astype(int)
    key=bx*100000+by; uk,inv=np.unique(key,return_inverse=True); ns=len(uk)
    Xb=np.zeros((ns,X.shape[1])); cxb=np.zeros(ns);cyb=np.zeros(ns);cnt=np.zeros(ns)
    np.add.at(Xb,inv,X); np.add.at(cxb,inv,cx); np.add.at(cyb,inv,cy); np.add.at(cnt,inv,1)
    cxb/=cnt; cyb/=cnt
    umi=Xb.sum(1); ok=umi>=depth
    Xb,cxb,cyb=Xb[ok],cxb[ok],cyb[ok]
    Rfull=geometry_R(cxb,cyb)   # R is computed once on the full pseudospot set, giving a stable centerline
    return Xb,cxb,cyb,Rfull

def decode_I(Xb,cxb,cyb,Rsub,depth,seed):
    rng=np.random.default_rng(seed)
    # depth-thin
    Xt=np.zeros_like(Xb)
    for i in range(Xb.shape[0]):
        tot=Xb[i].sum()
        Xt[i]=Xb[i] if tot<=depth else rng.multinomial(int(depth),Xb[i]/tot)
    lib=Xt.sum(1,keepdims=True); lib[lib==0]=1
    Xn=np.log1p(Xt/lib*np.median(lib))
    pos=(Rsub-Rsub.min())/(np.ptp(Rsub)+1e-9)
    xq=np.quantile(cxb,np.linspace(0,1,NB+1)); blk=np.clip(np.digitize(cxb,xq[1:-1]),0,NB-1)
    yhat=np.full(len(pos),np.nan)
    for f in range(NB):
        tr=blk!=f; te=blk==f
        if tr.sum()<10 or te.sum()<3: continue
        p=PCA(n_components=min(30,Xn.shape[1],tr.sum()-1),random_state=seed).fit(Xn[tr])
        m=Ridge(alpha=1.0).fit(p.transform(Xn[tr]),pos[tr]); yhat[te]=m.predict(p.transform(Xn[te]))
    ok=~np.isnan(yhat); var=np.var(pos[ok]); mse=np.mean((pos[ok]-yhat[ok])**2)
    I=0.5*np.log2(var/mse) if mse>0 and var>0 else np.nan
    return I

def run_arm():
  rows=[]
  with threadpool_limits(limits=1):
    for stage in ["GW8","GW10","GW12","GW16"]:
      try:
        Xm,gm,cxm,cym,ctm=load_merfish(stage)
        Xs,gs,cxs,cys=load_stereo(stage)
        Xmb,cxmb,cymb,Rm=pseudospots(cxm,cym,Xm,SCALE,DEPTH)
        Xsb,cxsb,cysb,Rs=pseudospots(cxs,cys,Xs,SCALE_STEREO,DEPTH)
        nm=Xmb.shape[0]; nsp=Xsb.shape[0]; nmatch=min(nm,nsp)
        if nmatch<N_MIN:
            rows.append(dict(stage=stage,n_mer=nm,n_stereo=nsp,n_match=nmatch,n_genes=Xm.shape[1],gene_layer=GENESET,
                I_mer_med=np.nan,I_stereo_med=np.nan,dI_med=np.nan,dI_lo=np.nan,dI_hi=np.nan,pct_pos=np.nan,nboot=0))
            print(f"{stage}: SKIP nmatch={nmatch}<{N_MIN}",flush=True); continue
        dIs=[]; Ims=[]; Iss=[]
        rng=np.random.default_rng(0)
        for b in range(B):
            im=rng.choice(nm,nmatch,replace=False); isx=rng.choice(nsp,nmatch,replace=False)
            Im=decode_I(Xmb[im],cxmb[im],cymb[im],Rm[im],DEPTH,b)
            Is=decode_I(Xsb[isx],cxsb[isx],cysb[isx],Rs[isx],DEPTH,b)
            if not(np.isnan(Im) or np.isnan(Is)): dIs.append(Im-Is); Ims.append(Im); Iss.append(Is)
        dIs=np.array(dIs); Ims=np.array(Ims); Iss=np.array(Iss)
        rows.append(dict(stage=stage,n_mer=nm,n_stereo=nsp,n_match=nmatch,n_genes=len(gm),gene_layer=GENESET,
            I_mer_med=round(float(np.median(Ims)),4),I_stereo_med=round(float(np.median(Iss)),4),
            dI_med=round(float(np.median(dIs)),4),dI_lo=round(float(np.percentile(dIs,2.5)),4),
            dI_hi=round(float(np.percentile(dIs,97.5)),4),pct_pos=round(float((dIs>0).mean()),3),nboot=len(dIs)))
        print(rows[-1],flush=True)
      except Exception as e:
        print(f"{stage}: ERROR {type(e).__name__}: {e}",flush=True)
        rows.append(dict(stage=stage,n_mer=-1,n_stereo=-1,n_match=-1,n_genes=-1,gene_layer=GENESET,
            I_mer_med=np.nan,I_stereo_med=np.nan,dI_med=np.nan,dI_lo=np.nan,dI_hi=np.nan,pct_pos=np.nan,nboot=0))

  return rows

ALL=[]
for _layer in ("shared","native"):
    GENESET=_layer
    print("=== gene set:",_layer,"===",flush=True)
    ALL+=run_arm()
df=pd.DataFrame(ALL)
df.to_csv(f"{RES_}/fig3_xtech_genelayer.csv",index=False)
print("WROTE fig3_xtech_genelayer.csv",df.shape)
