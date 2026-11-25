import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
# random_state=0 is pinned here for reproducible PCA fits

import numpy as np, pandas as pd, scipy.io as sio, os, math, json
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
rng=np.random.default_rng(1)
base=DATA_
order=["Fibroblast_Pia","Fibroblast_Arachnoid","Fibroblast_Dura"]
exec(open(f"{HERE_}/multiGW_scan.py").read().split("UMPP=0.5")[0])  # reuse the loader and radial-coordinate helper functions defined in multiGW_scan.py

UMPP=0.5; SCALES=[50,100,200,400,800]; DEPTHS=[16,32,64,128,256,512]; N_MIN=100
BLOCK_UM=800; NBOOT=500
def build_oof(gw):
    counts,coords,fib=load_stage(gw)
    r=radial_coord_full(coords[:,0],coords[:,1])
    lo,hi=np.percentile(r[fib],[3,90]); sel=fib&(r>=lo)&(r<=hi)
    xs,ys=coords[sel,0],coords[sel,1]; Cs=counts[sel]; rs=r[sel]
    lo2,hi2=np.percentile(rs,[3,97]); T_um=abs(hi2-lo2)*UMPP
    out={}
    for s_um in SCALES:
        step=s_um/UMPP
        gx=np.floor((xs-xs.min())/step).astype(int); gy=np.floor((ys-ys.min())/step).astype(int)
        key=gx*100000+gy; uk,inv=np.unique(key,return_inverse=True); nps=len(uk)
        agg=np.zeros((nps,Cs.shape[1])); Rp=np.zeros(nps); cx=np.zeros(nps); cy=np.zeros(nps)
        for i in range(nps):
            m=inv==i; agg[i]=Cs[m].sum(0); Rp[i]=rs[m].mean(); cx[i]=xs[m].mean(); cy[i]=ys[m].mean()
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

rows=[]
for gw in ["GW6","GW8","GW10","GW12","GW16"]:
    oof,T_um=build_oof(gw); cells=list(oof.keys())
    if not cells: print(gw,"no cells"); continue
    # block bootstrap: resample spatial blocks, recompute I per cell, derive effect sizes
    for bi in range(NBOOT):
        # E_depth: I(2d,s)-I(d,s); E_scale: I(d,2s)-I(d,s)
        for (s,d) in cells:
            o=oof[(s,d)]; blk=(np.floor((o["cx"]-o["cx"].min())/(BLOCK_UM/UMPP)).astype(int)*1000+
                                np.floor((o["cy"]-o["cy"].min())/(BLOCK_UM/UMPP)).astype(int))
            ub=np.unique(blk); samp=rng.choice(ub,len(ub),replace=True)
            idx=np.concatenate([np.where(blk==b)[0] for b in samp])
            I0=I_from(o["pos"],o["yhat"],idx)
            if (s,2*d) in oof:
                o2=oof[(s,2*d)]; blk2=(np.floor((o2["cx"]-o2["cx"].min())/(BLOCK_UM/UMPP)).astype(int)*1000+
                                       np.floor((o2["cy"]-o2["cy"].min())/(BLOCK_UM/UMPP)).astype(int))
                idx2=np.concatenate([np.where(blk2==b)[0] for b in samp if (blk2==b).any()])
                if len(idx2)>10: rows.append(dict(stage=gw,kind="E_depth",scale_um=s,depth=d,boot=bi,val=I_from(o2["pos"],o2["yhat"],idx2)-I0))
            if (2*s,d) in oof:
                o2=oof[(2*s,d)]; blk2=(np.floor((o2["cx"]-o2["cx"].min())/(BLOCK_UM/UMPP)).astype(int)*1000+
                                       np.floor((o2["cy"]-o2["cy"].min())/(BLOCK_UM/UMPP)).astype(int))
                idx2=np.concatenate([np.where(blk2==b)[0] for b in samp if (blk2==b).any()])
                if len(idx2)>10: rows.append(dict(stage=gw,kind="E_scale",scale_um=s,depth=d,boot=bi,val=I_from(o2["pos"],o2["yhat"],idx2)-I0))
    print(gw,"done cells=",len(cells),flush=True)
bdf=pd.DataFrame(rows)
bdf.to_csv(f"{RES_}/multiGW_bootstrap.csv",index=False)
# summary per stage
summ=[]
for gw in ["GW6","GW8","GW10","GW12","GW16"]:
    for kind in ["E_depth","E_scale"]:
        v=bdf[(bdf.stage==gw)&(bdf.kind==kind)&(bdf.depth>=64 if kind=="E_depth" else bdf.scale_um>=200)].val
        if len(v)==0: continue
        summ.append(dict(stage=gw,kind=kind,median=round(v.median(),4),
                         lo=round(np.percentile(v,2.5),4),hi=round(np.percentile(v,97.5),4),
                         pct_pos=round((v>0).mean(),3),pct_neg=round((v<0).mean(),3),n=len(v)))
sdf=pd.DataFrame(summ); sdf.to_csv(f"{RES_}/multiGW_effects_by_stage.csv",index=False)
print("BOOT_DONE"); print(sdf.to_string(index=False))
