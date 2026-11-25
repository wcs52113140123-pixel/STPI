"""Null-model test for the tangential (arclength) null hypothesis (Fig. 1d).

Null device: pseudospots are ordered by the tangential coordinate T (arclength along the
centreline); the expression matrix X is cyclically shifted along that order by a random
offset in [0.1n, 0.9n), while the position target R is kept fixed. Positional information is
estimated with checkerboard cross-validated decoding, over 200 null draws. Pseudospots use a
50 um cell size and a 512-read depth; seed 0.

Writes results to the path given as the first command-line argument, or to
results/tangential_null_draws.npz by default.
"""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
import sys

import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[_v]="1"
import numpy as np, pandas as pd, scipy.io as sio, time
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
base=DATA_; UMPP=0.5
order=["Fibroblast_Pia","Fibroblast_Arachnoid","Fibroblast_Dura"]
def load_stage(gw):
    if gw=="GW16":
        d=f"{DATA_}/gaston_gw16"; counts=np.load(f"{d}/counts_mat.npy"); coords=np.load(f"{d}/coords_mat.npy")
        clusters=np.load(f"{d}/clusters_kept.npy",allow_pickle=True)
        keep=counts.sum(1)>0; counts,coords,clusters=counts[keep],coords[keep],clusters[keep]; fib=np.isin(clusters,order)
    elif gw=="GW6":
        m=pd.read_csv(f"{DATA_}/gw6_annotated/expr/GW6_Stereo_R1_meta.csv")
        c=sio.mmread(f"{DATA_}/gw6_annotated/expr/GW6_Stereo_R1_counts.mtx"); counts=np.asarray(c.todense()).T
        coords=m[["x","y"]].values.astype(float); ct=m["celltype"].values; fib=np.isin(ct,["Fibro-pia-pro","Fibro-others"])
    else:
        d=f"{DATA_}/gaston_multi/{gw}"; counts=np.asarray(sio.mmread(f"{d}/counts.mtx").T.todense(),dtype=np.float64)
        coords=np.loadtxt(f"{d}/coords.tsv"); clusters=np.array([l.strip() for l in open(f"{d}/clusters.txt")]); fib=np.isin(clusters,order)
    return counts,coords,fib
def centerline_tr(x,y,nbin=60,ksm=5):
    cx,cy=np.median(x),np.percentile(y,15); theta=np.arctan2(y-cy,x-cx); tb=pd.cut(theta,nbin,labels=False)
    cl=[]
    for b in range(nbin):
        mm=tb==b
        if mm.sum()<15: continue
        cl.append((np.median(theta[mm]),np.median(x[mm]),np.median(y[mm])))
    cl=np.array(cl); cl=cl[np.argsort(cl[:,0])]
    sm=lambda a:np.convolve(a,np.ones(ksm)/ksm,mode="same"); clx,cly=sm(cl[:,1]),sm(cl[:,2])
    arclen=np.concatenate([[0],np.cumsum(np.hypot(np.diff(clx),np.diff(cly)))])
    tang=np.gradient(np.column_stack([clx,cly]),axis=0); tang/=np.linalg.norm(tang,axis=1,keepdims=True)+1e-9
    nrm=np.column_stack([-tang[:,1],tang[:,0]]); ro=np.column_stack([clx-cx,cly-cy]); ro/=np.linalg.norm(ro,axis=1,keepdims=True)+1e-9
    fl=np.sign((nrm*ro).sum(1)); fl[fl==0]=1; nrm*=fl[:,None]
    tree=cKDTree(np.column_stack([clx,cly])); _,idx=tree.query(np.column_stack([x,y]))
    r=((np.column_stack([x-clx[idx],y-cly[idx]]))*nrm[idx]).sum(1); t=arclen[idx]
    return t,r,arclen[-1]
def decode_I_cv(X,pos,cb,seed=0):
    yhat=np.zeros_like(pos)
    for f in [0,1]:
        tr=cb!=f; te=cb==f
        if tr.sum()<10 or te.sum()<5: return np.nan,np.nan
        p=PCA(n_components=min(30,X.shape[1],tr.sum()-1),random_state=seed).fit(X[tr])
        m=Ridge(alpha=1.0).fit(p.transform(X[tr]),pos[tr]); yhat[te]=m.predict(p.transform(X[te]))
    var=np.var(pos); mse=np.mean((pos-yhat)**2)
    I=0.5*np.log2(var/mse) if mse>0 and var>0 else np.nan
    return I,spearmanr(pos,yhat)[0]
def build_ps(gw,s_um=50,d=512):
    counts,coords,fib=load_stage(gw); x0,y0=coords[fib,0],coords[fib,1]; C=counts[fib]
    t,r,_=centerline_tr(x0,y0); lo,hi=np.percentile(r,[3,90]); keep=(r>=lo)&(r<=hi)
    x0,y0,C,t,r=x0[keep],y0[keep],C[keep],t[keep],r[keep]; step=s_um/UMPP
    gx=np.floor((x0-x0.min())/step).astype(int); gy=np.floor((y0-y0.min())/step).astype(int)
    key=gx*100000+gy; uk,inv=np.unique(key,return_inverse=True); nps=len(uk)
    agg=np.zeros((nps,C.shape[1])); Rp=np.zeros(nps); Tp=np.zeros(nps); cx=np.zeros(nps); cy=np.zeros(nps)
    for i in range(nps):
        mm=inv==i; agg[i]=C[mm].sum(0); Rp[i]=r[mm].mean(); Tp[i]=t[mm].mean(); cx[i]=x0[mm].mean(); cy[i]=y0[mm].mean()
    umi=agg.sum(1); ok=umi>=d
    if ok.sum()<100: return None
    A=agg[ok]; Rk=Rp[ok]; Tk=Tp[ok]; cxk=cx[ok]; cyk=cy[ok]
    rng=np.random.default_rng(0); At=np.zeros_like(A)
    for i in range(A.shape[0]):
        tot=A[i].sum(); At[i]=A[i] if tot<=d else rng.multinomial(d,A[i]/tot)
    lib=At.sum(1,keepdims=True); lib[lib==0]=1; X=np.log1p(At/lib*np.median(lib))
    pos=(Rk-Rk.min())/(np.ptp(Rk)+1e-9)
    bx=pd.cut(cxk,4,labels=False).astype(int); by=pd.cut(cyk,4,labels=False).astype(int); cb=(bx+by)%2
    return X,pos,cb,Tk
def full_null(gw,s_um=50,d=512,nnull=200,seed=0):
    ps=build_ps(gw,s_um,d)
    if ps is None: return None
    X,pos,cb,Tk=ps; Iobs,rhoobs=decode_I_cv(X,pos,cb,seed)
    oi=np.argsort(Tk); Xo=X[oi]; invo=np.argsort(oi); n=len(oi); rng=np.random.default_rng(seed); nulls=[]
    for _ in range(nnull):
        sh=rng.integers(int(0.1*n),int(0.9*n)); Xn=np.roll(Xo,sh,axis=0)[invo]
        In,_=decode_I_cv(Xn,pos,cb,seed)
        if not np.isnan(In): nulls.append(In)
    return Iobs,np.array(nulls)
t0=time.time(); out={}
for gw in ["GW6","GW8","GW10","GW12","GW16"]:
    r=full_null(gw); 
    if r:
        Iobs,nulls=r; out[f"{gw}_iobs"]=Iobs; out[f"{gw}_null"]=nulls
        print(gw,"Iobs=%.4f"%Iobs,"nnull=%d P=%.4f"%(len(nulls),(np.sum(nulls>=Iobs)+1)/(len(nulls)+1)),"[%.0fs]"%(time.time()-t0),flush=True)
np.savez(sys.argv[1] if len(sys.argv)>1 else f"{RES_}/tangential_null_draws.npz",**out)
print("SAVED",list(out.keys()))
