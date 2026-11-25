"""
MERFISH strict-grid coverage analysis.
Uses the same method as the Stereo-seq analysis: geometry-only tissue-interiority R
(zero expression input), spatial-block cross-validation with a buffer, and a decoder
information score I = 0.5*log2(Var(R)/MSE). Two gene-set layers are evaluated:
MERFISH-native (960 genes) and the Stereo-MERFISH shared set (265 genes), across a
depth x scale grid, gated by feasibility.
"""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[_v]="1"
import numpy as np, pandas as pd
from scipy.io import mmread
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

BASE=DATA_
EXPR=f"{DATA_}/gw6_annotated/expr"
DD=f"{DER_}"
RES=f"{RES_}"

SCALES=[50,100,200,400]          # um
DEPTHS=[16,32,64,128,256]        # transcripts/unit (capped at 256 for MERFISH)
N_MIN=100                        # minimum number of pseudospots required (same threshold used for Stereo-seq)
N_BLOCKS=5
BUFFER_MULT=1.0

def load_stage(stage):
    """cells x genes counts + coords (cx,cy) + celltype, on the annotated meninges subset."""
    sk={"GW6":"GW6_MERFISH_r0","GW8":"GW8_MERFISH","GW10":"GW10_MERFISH",
        "GW12":"GW12_MERFISH","GW16":"GW16_MERFISH"}[stage]
    M=mmread(f"{EXPR}/{sk}_counts.mtx").tocsr()
    genes=[l.strip() for l in open(f"{EXPR}/{sk}_genes.txt")]
    cells=np.array([l.strip() for l in open(f"{EXPR}/{sk}_cells.txt")])
    if M.shape[0]==len(genes): M=M.T          # -> cells x genes
    M=M.tocsr()
    cmap={c:i for i,c in enumerate(cells)}
    if stage=="GW6":
        meta=pd.read_csv(f"{EXPR}/GW6_MERFISH_r0_coords.csv"); meta['cell']=meta['cell'].astype(str)
        meta=meta[meta['cell'].isin(cmap)]
        idx=[cmap[c] for c in meta['cell']]
        lab=pd.read_csv(f"{DD}/merfish_meta_GW6_labels.csv"); lab['cell']=lab['cell'].astype(str)
        lmap=dict(zip(lab['cell'],lab['CellType']))
        ct=np.array([lmap.get(c,"NA") for c in meta['cell']])
        cx=meta['x'].values; cy=meta['y'].values
    else:
        meta=pd.read_csv(f"{DD}/merfish_meta_{stage}.csv"); meta['Cell_ID_2']=meta['Cell_ID_2'].astype(str)
        meta=meta[meta['Cell_ID_2'].isin(cmap)]
        idx=[cmap[c] for c in meta['Cell_ID_2']]
        cx=meta['cx'].values; cy=meta['cy'].values; ct=meta['CellType'].values
    X=np.asarray(M[idx].todense())
    return dict(X=X, genes=genes, cx=cx, cy=cy, ct=ct)

def geometry_R(cx,cy,nseg=12):
    """Geometry-only across-band radial coordinate (Pia to Dura normal direction), using no
    expression input. The meninges section is an elongated (possibly curved) band; the layer
    axis is the across-band normal. Fit a smooth centerline along the principal (tangential)
    axis via binned medians, then set R to the signed perpendicular offset from the local
    centerline, normalized to 0..1. This is the same centerline-normal construction used for
    Stereo-seq, applied uniformly to all stages. The direction (sign) is validated post hoc
    against the layer labels but is never used to define R. QC: |rho(R, Pia<Arach<Dura)| is
    about 0.32-0.51 across stages."""
    from sklearn.decomposition import PCA as _PCA
    pts=np.c_[cx,cy].astype(float)
    pr=_PCA(2).fit_transform(pts)
    t=pr[:,0]
    bins=np.quantile(t,np.linspace(0,1,nseg+1))
    bc=[]; bm=[]
    for i in range(nseg):
        s=(t>=bins[i])&(t<=bins[i+1])
        if s.sum()>=5: bc.append(np.median(t[s])); bm.append(np.median(pr[s,1]))
    bc=np.array(bc); bm=np.array(bm)
    center=np.interp(t,bc,bm)
    off=pr[:,1]-center
    R=(off-off.min())/(off.max()-off.min()+1e-9)
    return R

def spatial_blocks(pts):
    xq=np.quantile(pts[:,0],np.linspace(0,1,N_BLOCKS+1))
    return np.clip(np.digitize(pts[:,0],xq[1:-1]),0,N_BLOCKS-1)

def bin_pseudospots(cx,cy,X,scale):
    """Aggregate cells into scale-um square bins, producing pseudospot mean expression,
    centroid, and (later) mean R."""
    bx=np.floor(cx/scale).astype(int); by=np.floor(cy/scale).astype(int)
    key=bx*100000+by
    uk,inv=np.unique(key,return_inverse=True)
    ns=len(uk)
    Xb=np.zeros((ns,X.shape[1])); cxb=np.zeros(ns); cyb=np.zeros(ns); cnt=np.zeros(ns)
    np.add.at(Xb,inv,X); np.add.at(cxb,inv,cx); np.add.at(cyb,inv,cy); np.add.at(cnt,inv,1)
    Xb/=cnt[:,None]; cxb/=cnt; cyb/=cnt
    return Xb,cxb,cyb,cnt,inv

def downsample_depth(Xb,depth):
    """Multinomial-thin each pseudospot's mean-count vector to the target total 'depth'."""
    out=np.zeros_like(Xb)
    rng=np.random.default_rng(0)
    for i in range(Xb.shape[0]):
        tot=Xb[i].sum()
        if tot<=0: continue
        p=Xb[i]/tot
        out[i]=rng.multinomial(int(depth),p)
    return out

def decode_I(R,pts,X,mode,seed=0):
    n=len(R)
    if mode=='random':
        blk=np.random.default_rng(seed).integers(0,N_BLOCKS,n); buf=0.0
    else:
        blk=spatial_blocks(pts)
        dnn,_=cKDTree(pts).query(pts,k=2); buf=BUFFER_MULT*np.median(dnn[:,1])*3
    pred=np.full(n,np.nan)
    for kb in range(N_BLOCKS):
        te=np.where(blk==kb)[0]
        if len(te)<8: continue
        if mode=='random': tr=np.where(blk!=kb)[0]
        else:
            dmin,_=cKDTree(pts[te]).query(pts); tr=np.where((blk!=kb)&(dmin>buf))[0]
            if len(tr)<40: tr=np.where(blk!=kb)[0]
        nc=min(30,X.shape[1],len(tr)-1)
        if nc<2: continue
        pca=PCA(nc,random_state=0).fit(X[tr]); Zt=pca.transform(X[tr]); Ze=pca.transform(X[te])
        m=Zt.mean(0); sd=Zt.std(0)+1e-8
        rg=Ridge(alpha=1.0).fit((Zt-m)/sd,R[tr]); pred[te]=rg.predict((Ze-m)/sd)
    mk=~np.isnan(pred)
    if mk.sum()<20: return np.nan,np.nan,mk.sum()
    v=np.var(R[mk]); mse=np.mean((R[mk]-pred[mk])**2)
    I=0.5*np.log2(v/mse) if (mse>0 and v>0) else np.nan
    from scipy.stats import spearmanr
    rho=spearmanr(R[mk],pred[mk]).correlation
    return I,rho,mk.sum()

def run_all(gene_layer="native"):
    shared=set(l.strip() for l in open(f"{DD}/stereo_merfish_shared_genes.txt"))
    rows=[]
    for stage in ["GW6","GW8","GW10","GW12","GW16"]:
        d=load_stage(stage)
        genes=np.array(d['genes'])
        gmask=np.isin(genes,list(shared)) if gene_layer=="shared" else np.ones(len(genes),bool)
        X0=d['X'][:,gmask]; cx,cy=d['cx'],d['cy']
        Rcell=geometry_R(cx,cy)
        for scale in SCALES:
            Xb,cxb,cyb,cnt,inv=bin_pseudospots(cx,cy,X0,scale)
            keep=cnt>=1
            if keep.sum()<N_MIN:
                for depth in DEPTHS:
                    rows.append(dict(stage=stage,gene_layer=gene_layer,scale_um=scale,depth=depth,
                                     n=int(keep.sum()),I=np.nan,rho=np.nan,feasible=0))
                continue
            Rb=np.zeros(len(cxb)); np.add.at(Rb,inv,Rcell); Rb/=cnt; Rb=Rb[keep]
            Xk=Xb[keep]; pts=np.c_[cxb[keep],cyb[keep]]
            for depth in DEPTHS:
                Xd=downsample_depth(Xk,depth)
                for mode in ['random','spatial']:
                    I,rho,nn=decode_I(Rb,pts,Xd,mode)
                    rows.append(dict(stage=stage,gene_layer=gene_layer,scale_um=scale,depth=depth,cv=mode,
                                     n=int(keep.sum()),I=I,rho=rho,feasible=1))
    return pd.DataFrame(rows)

if __name__=="__main__":
    out=[]
    for gl in ["native","shared"]:
        df=run_all(gl); out.append(df); print(f"{gl}: {len(df)} rows, feasible={df.feasible.sum()}")
    full=pd.concat(out,ignore_index=True)
    full.to_csv(f"{RES}/merfish_grid_all.csv",index=False)
    print("WROTE merfish_grid_all.csv",full.shape)
