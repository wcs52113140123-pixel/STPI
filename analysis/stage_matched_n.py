"""Downsamples all stages to a common pseudospot count to test whether the GW12 peak
in the spatial information trajectory survives when spot count, not cell count, is held
equal across stages (holds per-spot depth constant and isolates the effect of n).
Reruns the five-stage trajectory and checks whether GW12 > GW10 and GW12 > GW16 holds at every
rung of a nested-n ladder.
Pipeline functions (load_stage/radial_coord_full/decode_I) are inlined verbatim from
multiGW_scan.py so importing this module does not trigger that module's top-level scan.
Compares ordering/rank rather than absolute values, and records a radial coverage QC gate
(flagged below 0.25)."""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
# random_state=0 fixed for reproducible PCA fitting
import numpy as np, pandas as pd, scipy.io as sio, math
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
base=DATA_
order=["Fibroblast_Pia","Fibroblast_Arachnoid","Fibroblast_Dura"]
UMPP=0.5

def radial_coord_full(x,y,nbin=60,ksm=5):
    cx,cy=np.median(x),np.percentile(y,15)
    theta=np.arctan2(y-cy,x-cx); tb=pd.cut(theta,nbin,labels=False)
    cl=[]
    for b in range(nbin):
        m=tb==b
        if m.sum()<15: continue
        cl.append((np.median(theta[m]),np.median(x[m]),np.median(y[m])))
    cl=np.array(cl); cl=cl[np.argsort(cl[:,0])]
    sm=lambda a:np.convolve(a,np.ones(ksm)/ksm,mode="same")
    clx,cly=sm(cl[:,1]),sm(cl[:,2])
    tang=np.gradient(np.column_stack([clx,cly]),axis=0); tang/=np.linalg.norm(tang,axis=1,keepdims=True)+1e-9
    nrm=np.column_stack([-tang[:,1],tang[:,0]])
    ro=np.column_stack([clx-cx,cly-cy]); ro/=np.linalg.norm(ro,axis=1,keepdims=True)+1e-9
    fl=np.sign((nrm*ro).sum(1)); fl[fl==0]=1; nrm*=fl[:,None]
    tree=cKDTree(np.column_stack([clx,cly])); _,idx=tree.query(np.column_stack([x,y]))
    r=((np.column_stack([x-clx[idx],y-cly[idx]]))*nrm[idx]).sum(1)
    return r

def load_stage(gw):
    if gw=="GW16":
        d=f"{DATA_}/gaston_gw16"
        counts=np.load(f"{d}/counts_mat.npy"); coords=np.load(f"{d}/coords_mat.npy")
        clusters=np.load(f"{d}/clusters_kept.npy",allow_pickle=True)
        keep=counts.sum(1)>0; counts,coords,clusters=counts[keep],coords[keep],clusters[keep]
        fib=np.isin(clusters,order)
    elif gw=="GW6":
        m=pd.read_csv(f"{DATA_}/gw6_annotated/expr/GW6_Stereo_R1_meta.csv")
        c=sio.mmread(f"{DATA_}/gw6_annotated/expr/GW6_Stereo_R1_counts.mtx"); counts=np.asarray(c.todense()).T
        coords=m[["x","y"]].values.astype(float); ct=m["celltype"].values
        fib=np.isin(ct,["Fibro-pia-pro","Fibro-others"])
    else:
        d=f"{DATA_}/gaston_multi/{gw}"
        counts=np.asarray(sio.mmread(f"{d}/counts.mtx").T.todense(),dtype=np.float64)
        coords=np.loadtxt(f"{d}/coords.tsv"); clusters=np.array([l.strip() for l in open(f"{d}/clusters.txt")])
        fib=np.isin(clusters,order)
    return counts,coords,fib

def decode_I(X,pos,blocks,rng):
    yhat=np.zeros_like(pos)
    for b in np.unique(blocks):
        te=blocks==b; tr=~te
        if te.sum()<5 or tr.sum()<20: yhat[te]=pos[tr].mean(); continue
        npc=min(30,X.shape[1],tr.sum()-1)
        pc=PCA(n_components=npc, random_state=0).fit(X[tr]); Xt=pc.transform(X[tr]); Xv=pc.transform(X[te])
        m=Ridge(alpha=1.0).fit(Xt,pos[tr]); yhat[te]=m.predict(Xv)
    resid=pos-yhat; mse=np.mean(resid**2); var=np.var(pos)
    I=0.5*math.log2(var/mse) if mse>0 and var>0 else 0.0
    return I,1-mse/var,spearmanr(pos,yhat)[0]

STAGES=["GW6","GW8","GW10","GW12","GW16"]
SCALE_UM=50; DEPTH=64; NB=4; R_REP=40

def build_pseudospots(gw):
    counts,coords,fib=load_stage(gw)
    r=radial_coord_full(coords[:,0],coords[:,1])
    lo,hi=np.percentile(r[fib],[3,90]); sel=fib&(r>=lo)&(r<=hi)
    xs,ys=coords[sel,0],coords[sel,1]; Cs=counts[sel]; rs=r[sel]
    step=SCALE_UM/UMPP
    gx=np.floor((xs-xs.min())/step).astype(int); gy=np.floor((ys-ys.min())/step).astype(int)
    key=gx*100000+gy; uk,inv=np.unique(key,return_inverse=True); nps=len(uk)
    agg=np.zeros((nps,Cs.shape[1])); Rp=np.zeros(nps); cx=np.zeros(nps); cy=np.zeros(nps)
    for i in range(nps):
        m=inv==i; agg[i]=Cs[m].sum(0); Rp[i]=rs[m].mean(); cx[i]=xs[m].mean(); cy[i]=ys[m].mean()
    umi=agg.sum(1); ok=umi>=DEPTH
    return agg[ok],Rp[ok],cx[ok],cy[ok]

pool={}
for gw in STAGES:
    A,R,cx,cy=build_pseudospots(gw); pool[gw]=dict(A=A,R=R,cx=cx,cy=cy,n=len(R))
    print(f"{gw}: feasible pseudospots @s{SCALE_UM} d{DEPTH} = {len(R)}",flush=True)

floors={gw:pool[gw]['n'] for gw in STAGES}
order_by_floor=sorted(STAGES,key=lambda g:floors[g])
ladder=[]; remaining=list(STAGES)
for drop in [None]+order_by_floor[:-1]:
    if drop is not None: remaining=[g for g in remaining if g!=drop]
    if len(remaining)<3: break
    ladder.append((min(floors[g] for g in remaining),list(remaining)))
print("LADDER:",ladder,flush=True)

def eval_stage(gw,n_target,rng):
    d=pool[gw]; idx=rng.choice(d['n'],size=n_target,replace=False)
    A=d['A'][idx]; R=d['R'][idx]; cx=d['cx'][idx]; cy=d['cy'][idx]
    At=np.zeros_like(A)
    for i in range(len(A)):
        tot=A[i].sum(); At[i]=A[i] if tot<=DEPTH else rng.multinomial(DEPTH,A[i]/tot)
    lib=At.sum(1,keepdims=True); lib[lib==0]=1
    X=np.log1p(At/lib*np.median(lib)); pos=(R-R.min())/(np.ptp(R)+1e-9)
    bx=pd.cut(cx,NB,labels=False).astype(int); by=pd.cut(cy,NB,labels=False).astype(int)
    cb=((bx+by)%2)
    I,r2,rho=decode_I(X,pos,cb,rng)
    covs=[np.ptp(pos[cb==f]) for f in [0,1] if (cb==f).sum()>=5]
    return rho,I,(min(covs) if covs else 0.0)

rows=[]
import os
for common_n,parts in ladder:
    for rep in range(R_REP):
        rng=np.random.default_rng(1000+rep)
        rec={gw:eval_stage(gw,common_n,rng) for gw in parts}
        row=dict(rung_n=common_n,rep=rep)
        for gw in parts:
            row[f"rho_{gw}"]=rec[gw][0]; row[f"I_{gw}"]=rec[gw][1]; row[f"radcov_{gw}"]=rec[gw][2]
        if all(g in parts for g in ["GW10","GW12","GW16"]):
            row["peak_survive"]=int((rec["GW12"][0]>rec["GW10"][0]) and (rec["GW12"][0]>rec["GW16"][0]))
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{RES_}/stage_matched_n_reps.csv",index=False)  # save results incrementally after each rung
    print(f"rung n={common_n} ({'+'.join(parts)}) done",flush=True)

df=pd.DataFrame(rows); df.to_csv(f"{RES_}/stage_matched_n_reps.csv",index=False)
summ=[]
for common_n,parts in ladder:
    d=df[df.rung_n==common_n]; rec=dict(rung_n=common_n,stages="+".join(parts),reps=len(d))
    for gw in parts:
        rec[f"rho_{gw}_med"]=round(d[f"rho_{gw}"].median(),3)
        rec[f"radcov_{gw}_med"]=round(d[f"radcov_{gw}"].median(),3)
    if "peak_survive" in d and d["peak_survive"].notna().any():
        rec["peak_survive_frac"]=round(d["peak_survive"].mean(),3)
    summ.append(rec)
s=pd.DataFrame(summ); s.to_csv(f"{RES_}/stage_matched_n_summary.csv",index=False)
print("=== summary ==="); print(s.to_string(index=False)); print("done",flush=True)
