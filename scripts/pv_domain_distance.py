"""PV domain distance: Luxembourg source PV vs Konstanz target PV."""
import numpy as np, pandas as pd
from scipy.stats import wasserstein_distance
W = ["shortwave_radiation","direct_radiation","diffuse_radiation","cloud_cover","temperature_2m","wind_speed_10m"]
src = pd.read_csv("Data/pv_target/lux_pv_source_features.csv")
tgt = pd.read_csv("Data/pv_target/konstanz_pv_features.csv")
def wn(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);s=np.std(np.concatenate([a,b])) or 1
    return wasserstein_distance(a/s,b/s)
print(f"{'variable':22s}{'W_norm':>8s}   LU(mean+-std)        Konstanz(mean+-std)")
for c in ["pv_kw"]+W:
    a,b=src[c].to_numpy(float),tgt[c].to_numpy(float)
    print(f"{c:22s}{wn(a,b):8.3f}   {a.mean():7.2f}+-{a.std():6.2f}   {b.mean():7.2f}+-{b.std():6.2f}")
def mmd2(X,Y,n=3000,seed=0):
    r=np.random.default_rng(seed);X=X[r.choice(len(X),min(n,len(X)),False)];Y=Y[r.choice(len(Y),min(n,len(Y)),False)]
    Z=np.vstack([X,Y]);d2=np.sum((Z[:500,None]-Z[None,:500])**2,-1);g=1/(np.median(d2[d2>0])+1e-9)
    k=lambda A,B:np.exp(-g*np.sum((A[:,None]-B[None])**2,-1));return float(k(X,X).mean()+k(Y,Y).mean()-2*k(X,Y).mean())
Wm=np.vstack([src[W].to_numpy(float),tgt[W].to_numpy(float)]);mu,sd=Wm.mean(0),Wm.std(0)+1e-9
print(f"\nweather MMD^2 = {mmd2((src[W].to_numpy(float)-mu)/sd,(tgt[W].to_numpy(float)-mu)/sd):.4f}")
print(f"pv_kw scale ratio (LU/Konstanz max) = {src.pv_kw.max()/max(tgt.pv_kw.max(),1e-9):.1f}x")
