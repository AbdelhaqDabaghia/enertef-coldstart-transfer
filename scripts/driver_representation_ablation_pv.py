"""
Representation ablation (PV, Luxembourg -> Konstanz): the decisive test of the
domain-informed axis. Here the domain-native representation is the CLEAR-SKY INDEX
(power / clear-sky envelope) -- far richer than EV's per-unit (a single scalar),
because the envelope is a per-(day-of-year, time-of-day) astronomical structure.
Question: does clear-sky beat the blind per-instance SOTA (RevIN), unlike per-unit
did on EV?

Same protocol as the EV ablation: hold everything fixed except the pv_kw / target
representation; train a PV source model FROM SCRATCH in each representation and
transfer (B0 scratch vs B3 warm) to Konstanz at N=30, K seeds; evaluate in kW.

Representations:
  STATIC GLOBAL (source-fit):   minmax | zscore | robust | log
  PER-INSTANCE / ADAPTIVE:      revin  | san_style   (labelled: our reimplementation)
  DOMAIN-INFORMED:              clear_sky  (train-only envelope, no holdout leak)

PV has a SINGLE power channel (pv_kw, seq index 0) and NO power in X_next.
Envelope = 95th-pct power per (10-day doy bin x 15-min tod), fit on TRAINING rows
only, applied to all rows (fallback = training median). Everything evaluated in kW.

Writes Data/results/representation_ablation_pv.csv (incremental + resume).
"""
import os
import numpy as np, pandas as pd, tensorflow as tf
from tensorflow import keras
from scipy.stats import wasserstein_distance, wilcoxon
from sklearn.preprocessing import MinMaxScaler
from coldstart_transfer.pv import (build_pv_model, PV_SEQ_FEATURES, PV_NEXT_EXO,
                                   PV_LOOKBACK as LB, PV_WEATHER)

for _g in tf.config.list_physical_devices("GPU"):
    try: tf.config.experimental.set_memory_growth(_g, True)
    except Exception: pass

PWI = PV_SEQ_FEATURES.index("pv_kw")                              # power channel = 0
EXO_SEQ = [i for i, c in enumerate(PV_SEQ_FEATURES) if c in PV_WEATHER]   # weather cols in seq
K_SEEDS = int(os.environ.get("ABL_SEEDS", "5"))
N_DAYS, HOLD = 30, 14
SAN_SLICES = 4                                                    # 96 / 24 = 4 six-hour slices
SRC_DAYS = 150
OUT_CSV = os.environ.get("ABL_OUT", "Data/results/representation_ablation_pv.csv")
REP_SUBSET = [r for r in os.environ.get("ABL_REPS", "").split(",") if r]

src = pd.read_csv("Data/pv_target/lux_pv_source_features.csv")
tgt = pd.read_csv("Data/pv_target/konstanz_pv_features.csv")
exo_scaler = MinMaxScaler().fit(src[PV_WEATHER].to_numpy(float))


# ---------------- clear-sky envelope (train-only) ----------------
def env_key(df):
    ts = pd.to_datetime(df.timestamp, utc=True)
    return (ts.dt.dayofyear // 10).astype(int).astype(str) + "_" + (ts.dt.hour*4 + ts.dt.minute//15).astype(str)

def fit_envelope(df, fit_idx):
    k = env_key(df)
    sub = df.iloc[fit_idx].assign(k=k.iloc[fit_idx])
    q = sub.groupby("k")["pv_kw"].quantile(0.95).clip(lower=0.05)
    fb = max(float(sub["pv_kw"].median()), 0.05)
    return np.maximum(k.map(q).fillna(fb).to_numpy(float), 0.05)


def scale_exo(df):
    """weather MinMax (source-fit); time sin/cos left as-is (already bounded)."""
    seq = df[PV_SEQ_FEATURES].to_numpy(np.float32).copy()
    nxt = df[PV_NEXT_EXO].to_numpy(np.float32).copy()
    seq[:, EXO_SEQ] = exo_scaler.transform(seq[:, EXO_SEQ])
    nxt_w = [i for i, c in enumerate(PV_NEXT_EXO) if c in PV_WEATHER]
    nxt[:, nxt_w] = exo_scaler.transform(df[PV_WEATHER].to_numpy(float))
    return seq, nxt


def windows_from(df, env_full=None):
    """Build (Xs kW-power, Xn, y_kW, env_seq[N,LB], env_pred[N]) from a frame slice.
    env_full aligned to df rows; env_seq/env_pred returned only if env_full given."""
    seq, nxt = scale_exo(df)
    ykw = df["pv_kw"].to_numpy(np.float32)
    n = len(df) - LB
    Xs = np.zeros((n, LB, len(PV_SEQ_FEATURES)), np.float32)
    Xn = np.zeros((n, len(PV_NEXT_EXO)), np.float32)
    y = np.zeros(n, np.float32)
    es = np.zeros((n, LB), np.float32) if env_full is not None else None
    ep = np.zeros(n, np.float32) if env_full is not None else None
    for i in range(n):
        Xs[i] = seq[i:i+LB]; Xn[i] = nxt[i+LB]; y[i] = ykw[i+LB]
        if env_full is not None:
            es[i] = env_full[i:i+LB]; ep[i] = env_full[i+LB]
    return Xs, Xn, y, es, ep


# ---------------- representations (window-level; power channel = PWI) ----------------
class GlobalRep:
    def __init__(self, name, fit_fn, fwd, inv):
        self.name, self.fit_fn, self.fwd, self.inv, self.p = name, fit_fn, fwd, inv, None
    def fit(self, sp): self.p = self.fit_fn(sp)
    def apply(self, Xs, Xn, y):
        Xs2 = Xs.copy()
        Xs2[:, :, PWI] = self.fwd(Xs2[:, :, PWI], self.p)
        return Xs2, Xn, self.fwd(y, self.p), (lambda pred: self.inv(pred, self.p))

class RevINRep:
    name = "revin"
    def apply(self, Xs, Xn, y):
        mu = Xs[:, :, PWI].mean(1); sg = Xs[:, :, PWI].std(1) + 1e-6
        Xs2 = Xs.copy(); Xs2[:, :, PWI] = (Xs2[:, :, PWI] - mu[:, None]) / sg[:, None]
        return Xs2, Xn, (y - mu) / sg, (lambda pred, _m=mu, _s=sg: pred * _s + _m)

class SANStyleRep:
    name = "san_style"
    def __init__(self): self.wmu = self.wsg = None
    def _stats(self, Xs):
        p = Xs.shape[1] // SAN_SLICES; ev = Xs[:, :, PWI]
        mus = np.stack([ev[:, s*p:(s+1)*p].mean(1) for s in range(SAN_SLICES)], 1)
        sgs = np.stack([ev[:, s*p:(s+1)*p].std(1) + 1e-6 for s in range(SAN_SLICES)], 1)
        return mus, sgs
    def fit_predictor(self, Xs, y):
        mus, sgs = self._stats(Xs); feat = np.c_[np.ones(len(Xs)), mus[:, -1], sgs[:, -1]]
        self.wmu = np.linalg.lstsq(feat, y, rcond=None)[0]
        self.wsg = np.linalg.lstsq(feat, sgs[:, -1], rcond=None)[0]
    def apply(self, Xs, Xn, y):
        mus, sgs = self._stats(Xs); p = Xs.shape[1] // SAN_SLICES; Xs2 = Xs.copy()
        for s in range(SAN_SLICES):
            sl = slice(s*p, (s+1)*p)
            Xs2[:, sl, PWI] = (Xs2[:, sl, PWI] - mus[:, s][:, None]) / sgs[:, s][:, None]
        feat = np.c_[np.ones(len(Xs)), mus[:, -1], sgs[:, -1]]
        mp = feat @ self.wmu; sp = np.abs(feat @ self.wsg) + 1e-6
        return Xs2, Xn, (y - mp) / sp, (lambda pred, _m=mp, _s=sp: pred * _s + _m)


def global_reps():
    return [
        GlobalRep("minmax", lambda p: dict(mx=float(np.quantile(p, .999)) + 1e-6),
                  lambda x, q: x / q["mx"], lambda x, q: x * q["mx"]),
        GlobalRep("zscore", lambda p: dict(mu=float(p.mean()), sd=float(p.std()) + 1e-6),
                  lambda x, q: (x - q["mu"]) / q["sd"], lambda x, q: x * q["sd"] + q["mu"]),
        GlobalRep("robust", lambda p: dict(md=float(np.median(p)),
                  iqr=float(np.quantile(p, .75) - np.quantile(p, .25)) + 1e-6),
                  lambda x, q: (x - q["md"]) / q["iqr"], lambda x, q: x * q["iqr"] + q["md"]),
        GlobalRep("log", lambda p: dict(lm=float(np.log1p(np.clip(p, 0, None)).mean()),
                  ls=float(np.log1p(np.clip(p, 0, None)).std()) + 1e-6),
                  lambda x, q: (np.log1p(np.clip(x, 0, None)) - q["lm"]) / q["ls"],
                  lambda x, q: np.expm1(x * q["ls"] + q["lm"])),
    ]


def clear_sky_windows(Xs, Xn, y, es, ep):
    """csi transform: divide pv_kw channel by lookback envelope, target by pred env."""
    Xs2 = Xs.copy()
    Xs2[:, :, PWI] = np.clip(Xs2[:, :, PWI] / es, 0, 1.3)
    y2 = np.clip(y / ep, 0, 1.3)
    return Xs2, Xn, y2, (lambda pred, _e=ep: pred * _e)


def domain_distance(a, b):
    ra = a[np.random.default_rng(0).choice(len(a), min(4000, len(a)), False)]
    rb = b[np.random.default_rng(1).choice(len(b), min(4000, len(b)), False)]
    return float(wasserstein_distance(ra, rb))


def main():
    s_fit = np.arange(0, len(src) - 90*96)
    t_fit = np.arange(0, len(tgt) - HOLD*96)
    s_env_full = fit_envelope(src, s_fit)
    t_env_full = fit_envelope(tgt, t_fit)

    # source tail (train), target head (first N_DAYS train), target holdout
    a = max(0, len(src) - (SRC_DAYS*96 + LB))
    src_tail = src.iloc[a:].reset_index(drop=True)
    sXs, sXn, sy, s_es, s_ep = windows_from(src_tail, s_env_full[a:])
    th = tgt.iloc[:LB + N_DAYS*96].reset_index(drop=True)
    thXs, thXn, thy, th_es, th_ep = windows_from(th, t_env_full[:LB + N_DAYS*96])
    b = len(tgt) - (LB + HOLD*96)
    hd = tgt.iloc[b:].reset_index(drop=True)
    hXs, hXn, hy, h_es, h_ep = windows_from(hd, t_env_full[b:])
    print(f"[pv-abl] src_win={len(sy)} tgt_head={len(thy)} holdout={len(hy)} "
          f"src_pv_max={sy.max():.1f} tgt_pv_max={hy.max():.1f}", flush=True)

    if os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV); rows = prev.to_dict("records")
        done_cond = {(r["rep"], r["cond"]) for r in rows}
        done = {r for r in prev.rep.unique() if set(prev[prev.rep == r].cond) >= {"B0", "B3"}}
        print(f"[pv-abl] resuming; done: {sorted(done)}", flush=True)
    else:
        rows, done, done_cond = [], set(), set()
    flush = lambda: pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    rep_names = ["zscore", "robust", "log", "revin", "san_style", "clear_sky"]  # minmax dropped
    for name in rep_names:
        if (REP_SUBSET and name not in REP_SUBSET) or name in done:
            continue
        keras.backend.clear_session()
        # ---- transform source-train + target-head + holdout in this representation ----
        if name == "clear_sky":
            sXs_t, sXn_t, sy_t, _ = clear_sky_windows(sXs, sXn, sy, s_es, s_ep)
            tr = clear_sky_windows(thXs, thXn, thy, th_es, th_ep)
            ho = clear_sky_windows(hXs, hXn, hy, h_es, h_ep)
        elif name == "revin":
            rep = RevINRep()
            sXs_t, sXn_t, sy_t, _ = rep.apply(sXs, sXn, sy)
            tr = rep.apply(thXs, thXn, thy); ho = rep.apply(hXs, hXn, hy)
        elif name == "san_style":
            rep = SANStyleRep(); rep.fit_predictor(sXs, sy)
            sXs_t, sXn_t, sy_t, _ = rep.apply(sXs, sXn, sy)
            tr = rep.apply(thXs, thXn, thy); ho = rep.apply(hXs, hXn, hy)
        else:
            rep = next(r for r in global_reps() if r.name == name); rep.fit(sy)
            sXs_t, sXn_t, sy_t, _ = rep.apply(sXs, sXn, sy)
            tr = rep.apply(thXs, thXn, thy); ho = rep.apply(hXs, hXn, hy)
        tXs_tr, tXn_tr, ty_tr, _ = tr
        tXs_ho, tXn_ho, _, denorm_ho = ho
        dist = domain_distance(sy_t, ty_tr)

        # ---- source PV model trained once in this representation ----
        keras.utils.set_random_seed(0); srcm = build_pv_model(seed=0)
        srcm.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.Huber(0.5))
        srcm.fit([sXs_t, sXn_t], sy_t, epochs=12, batch_size=128, verbose=0)
        src_w = srcm.get_weights()

        for cond in ("B0", "B3"):
            if (name, cond) in done_cond:
                continue
            errs = []
            for seed in range(K_SEEDS):
                keras.backend.clear_session()
                keras.utils.set_random_seed(seed); m = build_pv_model(seed=seed)
                if cond == "B3": m.set_weights(src_w)
                m.compile(optimizer=keras.optimizers.Adam(1e-4), loss=keras.losses.Huber(0.5))
                m.fit([tXs_tr, tXn_tr], ty_tr, epochs=10, batch_size=64, verbose=0)
                pred = m.predict([tXs_ho, tXn_ho], verbose=0).reshape(-1)
                pred_kw = np.clip(denorm_ho(pred), 0, None)
                rmse = float(np.sqrt(np.mean((pred_kw - hy) ** 2)))
                errs.append(rmse / (float(hy.mean()) + 1e-9))
            rows.append(dict(rep=name, cond=cond, dist_W=round(dist, 4),
                             nrmse_mean=round(float(np.mean(errs)), 4),
                             nrmse_std=round(float(np.std(errs)), 4),
                             rmse_kw_mean=round(float(np.mean([e * hy.mean() for e in errs])), 3),
                             seeds=";".join(f"{e:.4f}" for e in errs)))
            print(rows[-1], flush=True); flush()

    R = pd.DataFrame(rows); R.to_csv(OUT_CSV, index=False)
    b3 = R[R.cond == "B3"].sort_values("nrmse_mean").reset_index(drop=True)
    print(f"\n=== PV warm-start (B3) ranking, N=30, {K_SEEDS} seeds ===")
    print(b3[["rep", "nrmse_mean", "nrmse_std", "rmse_kw_mean", "dist_W"]].to_string(index=False))
    best = b3.iloc[0]["rep"]
    bs = np.array([float(x) for x in R[(R.rep == best) & (R.cond == "B3")].iloc[0]["seeds"].split(";")])
    print(f"\nWilcoxon paired vs best ({best}):")
    for _, r in b3.iloc[1:].iterrows():
        os_ = np.array([float(x) for x in R[(R.rep == r["rep"]) & (R.cond == "B3")].iloc[0]["seeds"].split(";")])
        try: p = wilcoxon(bs, os_).pvalue
        except Exception: p = float("nan")
        print(f"  {best} vs {r['rep']:9s}: p={p:.4f}")
    print("DONE")


if __name__ == "__main__":
    main()
