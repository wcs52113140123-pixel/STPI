import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
# random_state=0 is fixed throughout so that the down-sampling and cross-validation splits are reproducible across runs

import numpy as np, pandas as pd, scipy.io as sio, os, math
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
rng=np.random.default_rng(0)
base=DATA_
order=["Fibroblast_Pia","Fibroblast_Arachnoid","Fibroblast_Dura"]

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
    # checkerboard CV: predict held-out block, calibrated I
    yhat=np.zeros_like(pos)
    for b in np.unique(blocks):
        te=blocks==b; tr=~te
        if te.sum()<5 or tr.sum()<20: yhat[te]=pos[tr].mean(); continue
        npc=min(30,X.shape[1],tr.sum()-1)
        pc=PCA(n_components=npc, random_state=0).fit(X[tr])
        Xt=pc.transform(X[tr]); Xv=pc.transform(X[te])
        m=Ridge(alpha=1.0).fit(Xt,pos[tr]); yhat[te]=m.predict(Xv)
    resid=pos-yhat; mse=np.mean(resid**2); var=np.var(pos)
    I=0.5*math.log2(var/mse) if mse>0 and var>0 else 0.0
    r2=1-mse/var; rho=spearmanr(pos,yhat)[0]
    return I,r2,rho

UMPP=0.5  # 1 gem/bin50 unit = 0.5 um
SCALES_UM=[50,100,200,400,800]
DEPTHS=[16,32,64,128,256,512]
N_MIN=100
rows=[]; Tinfo={}
for gw in ["GW6","GW8","GW10","GW12","GW16"]:
    counts,coords,fib=load_stage(gw)
    r=radial_coord_full(coords[:,0],coords[:,1])
    lo,hi=np.percentile(r[fib],[3,90]); sel=fib&(r>=lo)&(r<=hi)
    xs,ys=coords[sel,0],coords[sel,1]; Cs=counts[sel]; rs=r[sel]
    # thickness T (um): radial normal span p3-p97
    lo2,hi2=np.percentile(rs,[3,97]); T_um=abs(hi2-lo2)*UMPP
    Tinfo[gw]=T_um
    pitch=np.median(cKDTree(np.column_stack([xs,ys])).query(np.column_stack([xs,ys]),k=2)[0][:,1])*UMPP
    for s_um in SCALES_UM:
        step=s_um/UMPP  # units
        gx=np.floor((xs-xs.min())/step).astype(int); gy=np.floor((ys-ys.min())/step).astype(int)
        key=gx*100000+gy
        uk,inv=np.unique(key,return_inverse=True)
        nps=len(uk)
        # aggregate counts + mean R per pseudospot
        agg=np.zeros((nps,Cs.shape[1])); Rp=np.zeros(nps); cx=np.zeros(nps); cy=np.zeros(nps)
        for i in range(nps):
            m=inv==i; agg[i]=Cs[m].sum(0); Rp[i]=rs[m].mean(); cx[i]=xs[m].mean(); cy[i]=ys[m].mean()
        umi=agg.sum(1)
        # feasibility: enough pseudospots with enough umi
        for d in DEPTHS:
            ok=umi>=d
            if ok.sum()<N_MIN: 
                rows.append(dict(stage=gw,scale_um=s_um,depth=d,n=int(ok.sum()),I=np.nan,R2=np.nan,rho=np.nan,rho_res=round(s_um/T_um,3),feasible=False))
                continue
            A=agg[ok]; Rk=Rp[ok]; cxk=cx[ok]; cyk=cy[ok]
            # thin to depth d (multinomial down-sample per pseudospot)
            At=np.zeros_like(A)
            for i in range(A.shape[0]):
                tot=A[i].sum()
                if tot<=d: At[i]=A[i]
                else:
                    p=A[i]/tot; At[i]=rng.multinomial(d,p)
            lib=At.sum(1,keepdims=True); lib[lib==0]=1
            X=np.log1p(At/lib*np.median(lib))
            pos=(Rk-Rk.min())/(np.ptp(Rk)+1e-9)
            # checkerboard blocks (4x4 spatial)
            nb=4; bx=pd.cut(cxk,nb,labels=False); by=pd.cut(cyk,nb,labels=False)
            blocks=(bx.astype(int))*nb+by.astype(int)
            # parity checkerboard: 2-fold by (bx+by)%2
            cb=((bx.astype(int)+by.astype(int))%2)
            try:
                I,r2,rho=decode_I(X,pos,cb,rng)
            except Exception as e:
                I,r2,rho=np.nan,np.nan,np.nan
            rows.append(dict(stage=gw,scale_um=s_um,depth=d,n=int(ok.sum()),I=round(I,4),R2=round(r2,4),rho=round(rho,4),rho_res=round(s_um/T_um,3),feasible=True))
    print(f"{gw}: T={T_um:.0f}um pitch={pitch:.0f}um done",flush=True)
df=pd.DataFrame(rows)
df.to_csv(f"{RES_}/multiGW_grid_all.csv",index=False)
import json; json.dump(Tinfo,open(f"{RES_}/multiGW_T.json","w"))
print("SCAN_DONE feasible cells:",int(df.feasible.sum()),"/",len(df))
print(df[df.feasible].groupby("stage").agg(rho_max=("rho","max"),I_max=("I","max"),n_cells=("feasible","sum")).to_string())
