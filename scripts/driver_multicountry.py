"""
Multi-country distance->gap study (#3). One LU per-unit PV source model, transfer
(B3 warm vs B0 scratch) to 9 European countries at N=90, and relate each country's
source->target domain distance (weather MMD + generation-shape Wasserstein, now
varying by location) to warm-start transfer error. Spearman across countries.
"""
import glob, os, numpy as np, pandas as pd, tensorflow as tf
from tensorflow import keras
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.preprocessing import MinMaxScaler
from coldstart_transfer.pv import build_pv_model, PV_SEQ_FEATURES, PV_NEXT_EXO, PV_LOOKBACK as LB, PV_WEATHER

BASE_LU = 751.2
lu = pd.read_csv("Data/pv_target/lux_pv_source_features.csv")
wxsc = MinMaxScaler().fit(lu[PV_WEATHER].to_numpy(float))

def prep(df, pv_is_perunit):
    d = pd.DataFrame({"timestamp": df.timestamp})
    d["pv_kw"] = df.pv_kw.to_numpy(float) if pv_is_perunit else df.pv_kw.to_numpy(float)/BASE_LU
    d[PV_WEATHER] = wxsc.transform(df[PV_WEATHER].to_numpy(float))
    for c in ["tod_sin","tod_cos","doy_sin","doy_cos"]: d[c] = df[c].to_numpy(float)
    return d

def windows(d):
    seq = d[PV_SEQ_FEATURES].to_numpy(np.float32); nxt = d[PV_NEXT_EXO].to_numpy(np.float32); yv = d.pv_kw.to_numpy(np.float32)
    n = len(d)-LB
    Xs=np.zeros((n,LB,len(PV_SEQ_FEATURES)),np.float32); Xn=np.zeros((n,len(PV_NEXT_EXO)),np.float32); y=np.zeros(n,np.float32)
    for i in range(n): Xs[i]=seq[i:i+LB]; Xn[i]=nxt[i+LB]; y[i]=yv[i+LB]
    return Xs,Xn,y

# LU per-unit source model (once)
lud = prep(lu, pv_is_perunit=False); sX,sN,sy = windows(lud); s_tr = slice(0,len(sy)-90*96)
keras.utils.set_random_seed(0); srcm = build_pv_model(seed=0)
srcm.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.Huber(0.5))
srcm.fit([sX[s_tr],sN[s_tr]], sy[s_tr], epochs=12, batch_size=128, verbose=0)
print("[multi] LU per-unit source trained")

def mmd2(X,Y,n=1500,seed=0):
    r=np.random.default_rng(seed); X=X[r.choice(len(X),min(n,len(X)),False)]; Y=Y[r.choice(len(Y),min(n,len(Y)),False)]
    Z=np.vstack([X,Y]); d2=np.sum((Z[:400,None]-Z[None,:400])**2,-1); g=1/(np.median(d2[d2>0])+1e-9)
    k=lambda A,B: np.exp(-g*np.sum((A[:,None]-B[None])**2,-1)); return float(k(X,X).mean()+k(Y,Y).mean()-2*k(X,Y).mean())

lu_w = wxsc.transform(lu[PV_WEATHER].to_numpy(float)); lu_pu = (lu.pv_kw/BASE_LU).to_numpy()
rows=[]
for path in sorted(glob.glob("Data/multicountry/*_features.csv")):
    cc = os.path.basename(path).replace("_features.csv","")
    df = pd.read_csv(path); d = prep(df, pv_is_perunit=True); Xs,Xn,y = windows(d)
    ho = slice(len(y)-14*96,len(y)); tr = slice(0,90*96)
    # distance
    cw = wxsc.transform(df[PV_WEATHER].to_numpy(float))
    wmmd = mmd2(lu_w, cw); wshape = wasserstein_distance(lu_pu, df.pv_kw.to_numpy(float))
    # transfer
    res={}
    for cond in ("B0","B3"):
        errs=[]
        for seed in (0,1):
            keras.utils.set_random_seed(seed); m=build_pv_model(seed=seed)
            if cond=="B3": m.set_weights(srcm.get_weights())
            m.compile(optimizer=keras.optimizers.Adam(1e-4), loss=keras.losses.Huber(0.5))
            m.fit([Xs[tr],Xn[tr]], y[tr], epochs=10, batch_size=64, verbose=0)
            p=np.clip(m.predict([Xs[ho],Xn[ho]],verbose=0).reshape(-1),0,None)
            errs.append(float(np.sqrt(np.mean((p-y[ho])**2)))/ (float(np.mean(y[ho]))+1e-9))
        res[cond]=np.mean(errs)
    rows.append(dict(country=cc, weather_mmd=round(wmmd,4), pv_shape_W=round(wshape,4),
                     b3_nrmse=round(res["B3"],4), b0_nrmse=round(res["B0"],4), improvement=round(res["B0"]/res["B3"],2)))
    print(rows[-1], flush=True)

R=pd.DataFrame(rows); R.to_csv("Data/results/multicountry.csv", index=False)
for xc in ("weather_mmd","pv_shape_W"):
    rho,p=spearmanr(R[xc], R.b3_nrmse); print(f"Spearman({xc}, warm-start nRMSE) = {rho:+.3f} (p={p:.3f}, n={len(R)})")
print("DONE")
