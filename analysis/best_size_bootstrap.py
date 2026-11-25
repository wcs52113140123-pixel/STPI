"""Bootstrap of the best pseudospot size at each stage (Fig S2 B).

For each stage the five-stage pipeline of multiGW_scan.py is rebuilt (same loader,
coordinate, QC crop, pseudospot binning, multinomial depth thinning and checkerboard
PCA + ridge decoder) and the out-of-fold predictions are kept. At the largest depth that
every pseudospot size reaches, spatial blocks of 800 um are resampled with replacement
NBOOT = 500 times; in each draw the size with the highest recoverability is the best size
s* (variables sstar). T is the radial thickness of the tissue band (T_um).

Writes
  results/best_size_boot_raw.json         best size of every draw, per stage (Fig S2 B)
  results/best_size_summary.csv           modal, median and 95% range of s*, and s*/T
  results/best_size_rawI_bootstrap.csv    recoverability per size for the first 200 draws
  results/best_size_spearman.json         Spearman rho between T and s* across stages
  results/multiGW_T_check.json            T per stage
"""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
# random_state=0 is fixed throughout so the bootstrap draws are reproducible

import numpy as np, pandas as pd, scipy.io as sio, os, math, json, sys
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
rng=np.random.default_rng(7)
base=DATA_
exec(open(f"{HERE_}/multiGW_scan.py").read().split("UMPP=0.5")[0])

UMPP=0.5; SCALES=[50,100,200,400,800]; DEPTHS=[16,32,64,128,256,512]; N_MIN=100
BLOCK_UM=800; NBOOT=500

def build_oof(gw):
    counts,coords,fib=load_stage(gw)
    r=radial_coord_full(coords[:,0],coords[:,1])
    lo,hi=np.percentile(r[fib],[3,90]); sel=fib&(r>=lo)&(r<=hi)
    xs,ys=coords[sel,0],coords[sel,1]; Cs=counts[sel].astype(np.float64); rs=r[sel]
    lo2,hi2=np.percentile(rs,[3,97]); T_um=abs(hi2-lo2)*UMPP
    G=Cs.shape[1]; out={}
    for s_um in SCALES:
        step=s_um/UMPP
        gx=np.floor((xs-xs.min())/step).astype(int); gy=np.floor((ys-ys.min())/step).astype(int)
        key=gx*100000+gy; uk,inv=np.unique(key,return_inverse=True); nps=len(uk)
        # vectorized aggregation
        agg=np.zeros((nps,G)); np.add.at(agg,inv,Cs)
        cnt=np.bincount(inv,minlength=nps).astype(float); cnt[cnt==0]=1
        Rp=np.bincount(inv,weights=rs,minlength=nps)/cnt
        cx=np.bincount(inv,weights=xs,minlength=nps)/cnt
        cy=np.bincount(inv,weights=ys,minlength=nps)/cnt
        umi=agg.sum(1)
        for d in DEPTHS:
            ok=umi>=d
            if ok.sum()<N_MIN: continue
            A=agg[ok]; Rk=Rp[ok]; cxk=cx[ok]; cyk=cy[ok]
            At=np.zeros_like(A)
            for i in range(A.shape[0]):
                tot=A[i].sum(); At[i]=A[i] if tot<=d else rng.multinomial(d,A[i]/tot)
            lib=At.sum(1,keepdims=True); lib[lib==0]=1
            X=np.log1p(At/lib*np.median(lib)); pos=(Rk-Rk.min())/(np.ptp(Rk)+1e-9)
            nb=4; bx=pd.cut(cxk,nb,labels=False).astype(int); by=pd.cut(cyk,nb,labels=False).astype(int)
            cb=(bx+by)%2
            yhat=np.zeros_like(pos)
            for b in np.unique(cb):
                te=cb==b; tr=~te
                if te.sum()<5 or tr.sum()<20: yhat[te]=pos[tr].mean(); continue
                npc=min(30,X.shape[1],tr.sum()-1); pc=PCA(n_components=npc, random_state=0).fit(X[tr])
                yhat[te]=Ridge(alpha=1.0).fit(pc.transform(X[tr]),pos[tr]).predict(pc.transform(X[te]))
            out[(s_um,d)]=dict(pos=pos,yhat=yhat,cx=cxk,cy=cyk)
    return out,T_um

def I_from(pos,yhat,idx):
    resid=(pos-yhat)[idx]; mse=np.mean(resid**2); var=np.var(pos[idx])
    return 0.5*math.log2(var/mse) if mse>0 and var>0 else 0.0

STAGES=["GW6","GW8","GW10","GW12","GW16"]
rawrows=[]; Tvals={}; sstar_boot={st:[] for st in STAGES}
for gw in STAGES:
    oof,T_um=build_oof(gw); Tvals[gw]=T_um
    cells=list(oof.keys()); fs=sorted(set(s for (s,d) in cells))
    depth_sets=[set(d for (s,d) in cells if s==s0) for s0 in fs]
    common=set.intersection(*depth_sets) if depth_sets else set()
    dref=max(common) if common else None
    print(gw,"feasible_scales",fs,"dref",dref,"T",round(T_um,1),flush=True)
    meta={}
    for (s,d) in cells:
        o=oof[(s,d)]
        blk=(np.floor((o["cx"]-o["cx"].min())/(BLOCK_UM/UMPP)).astype(int)*1000+
             np.floor((o["cy"]-o["cy"].min())/(BLOCK_UM/UMPP)).astype(int))
        meta[(s,d)]=blk
    for bi in range(NBOOT):
        Iscale={}
        for s in fs:
            if (s,dref) not in oof: continue
            o=oof[(s,dref)]; blk=meta[(s,dref)]
            ub=np.unique(blk); samp=rng.choice(ub,len(ub),replace=True)
            idx=np.concatenate([np.where(blk==b)[0] for b in samp])
            Ival=I_from(o["pos"],o["yhat"],idx); Iscale[s]=Ival
            if bi<200: rawrows.append(dict(stage=gw,scale_um=s,depth=dref,boot=bi,I=round(Ival,5)))
        if Iscale: sstar_boot[gw].append(max(Iscale,key=Iscale.get))
    # partial save after each stage
    pd.DataFrame(rawrows).to_csv(f"{RES_}/best_size_rawI_bootstrap.csv",index=False)
    json.dump({k:sstar_boot[k] for k in STAGES if sstar_boot[k]},
              open(f"{RES_}/best_size_boot_raw.json","w"))
    print(gw,"done nsstar",len(sstar_boot[gw]),flush=True)

json.dump(Tvals,open(f"{RES_}/multiGW_T_check.json","w"))
srows=[]
for gw in STAGES:
    arr=np.array(sstar_boot[gw])
    if len(arr)==0: continue
    vals,counts=np.unique(arr,return_counts=True); modal=vals[np.argmax(counts)]
    srows.append(dict(stage=gw,T_um=round(Tvals[gw],1),sstar_modal=int(modal),
        sstar_median=float(np.median(arr)),sstar_lo=float(np.percentile(arr,2.5)),
        sstar_hi=float(np.percentile(arr,97.5)),sT_modal=round(modal/Tvals[gw],3),
        sT_median=round(np.median(arr)/Tvals[gw],3),
        pmf=";".join(f"{int(v)}:{c/len(arr):.2f}" for v,c in zip(vals,counts))))
pd.DataFrame(srows).to_csv(f"{RES_}/best_size_summary.csv",index=False)
Tarr=np.array([Tvals[g] for g in STAGES])
cb=min(len(sstar_boot[g]) for g in STAGES)
rho_boot=[]
for bi in range(cb):
    sv=np.array([sstar_boot[g][bi] for g in STAGES])
    if len(np.unique(sv))>1: rho_boot.append(spearmanr(Tarr,sv)[0])
rho_boot=np.array(rho_boot)
res=dict(rho_median=float(np.median(rho_boot)),rho_lo=float(np.percentile(rho_boot,2.5)),
         rho_hi=float(np.percentile(rho_boot,97.5)),pct_positive=float((rho_boot>0).mean()),n_boot=len(rho_boot))
json.dump(res,open(f"{RES_}/best_size_spearman.json","w"))
print("best size per stage"); print(pd.DataFrame(srows).to_string(index=False))
print("Spearman rho (T, best size)",res); print("done",flush=True)
