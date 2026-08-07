"""
Representation ablation (EV, LU -> UK): does the OUTPUT/POWER representation govern
cold-start transfer, and where do domain-native reps sit relative to the 2024-2026
instance/adaptive-normalization SOTA?

We hold EVERYTHING fixed except how the 8 POWER channels (ev_kw + its lags/rolls)
and the target are represented. Exogenous channels (weather + time) use one common
MinMax scaler fit on the source, in every condition. Because each representation
changes the target space, the frozen production weights no longer apply; we train a
source model FROM SCRATCH in each representation and transfer it (B0 scratch vs B3
warm) to the UK at N=30, K seeds. Everything is evaluated back in kW.

Representations (three axes):
  STATIC GLOBAL (fit on source power):  minmax | zscore | robust | log
  PER-INSTANCE / ADAPTIVE (SOTA):       revin  | san_style
  DOMAIN-INFORMED (our contribution):   per_unit  (divide by each site's capacity)

For each we also measure the TARGET-VARIABLE domain distance in the transformed
space (1-Wasserstein between source and target transformed targets) -- the
mechanism knob: a representation that shrinks this distance should transfer better.

Honesty notes:
 * revin = RevIN WITHOUT the learnable affine (per-window standardisation of the
   power channels by the lookback mean/std of ev_kw; prediction de-normalised with
   the same stats). We drop the affine deliberately (contested in 2026 literature).
 * san_style = OUR reimplementation of SAN's idea (per-slice normalisation +
   a small learned predictor of the next-step local statistics), NOT the authors'
   code. Labelled as such everywhere.

Writes Data/results/representation_ablation_ev.csv.
"""
import os
import numpy as np, pandas as pd, tensorflow as tf
from tensorflow import keras
from scipy.stats import wasserstein_distance, wilcoxon
from sklearn.preprocessing import MinMaxScaler
from coldstart_transfer.model import build_ev_model
from coldstart_transfer.features import SEQ_FEATURES, NEXT_EXO

for _g in tf.config.list_physical_devices("GPU"):
    try: tf.config.experimental.set_memory_growth(_g, True)
    except Exception: pass

LB = 672
POWER = ("ev_kw", "lag_1", "lag_4", "lag_96", "lag_672",
         "roll_1h_mean", "roll_6h_mean", "roll_24h_mean")
PW_SEQ = [i for i, c in enumerate(SEQ_FEATURES) if c in POWER]     # 0..7 (incl ev_kw)
PW_NXT = [i for i, c in enumerate(NEXT_EXO) if c in POWER]         # lags/rolls (no ev_kw)
EXO_SEQ = [i for i in range(len(SEQ_FEATURES)) if i not in PW_SEQ]
EXO_NXT = [i for i in range(len(NEXT_EXO)) if i not in PW_NXT]
EVKW_SEQ = SEQ_FEATURES.index("ev_kw")                             # channel 0
K_SEEDS = int(os.environ.get("ABL_SEEDS", "5"))                   # override for more seeds
N_DAYS, HOLD = 30, 14
SAN_SLICES = 7                                                     # 672 / 96 = 7 daily slices
OUT_CSV = os.environ.get("ABL_OUT", "Data/results/representation_ablation_ev.csv")
REP_SUBSET = [r for r in os.environ.get("ABL_REPS", "").split(",") if r]  # empty = all

src = pd.read_csv("Data/lux_source_features.csv")
tgt = pd.read_csv("Data/uk_ev_features_full.csv")

# one common exogenous MinMax scaler (source-fit) shared by ALL representations
exo_scaler = MinMaxScaler().fit(src[[SEQ_FEATURES[i] for i in EXO_SEQ]].to_numpy(float))


def raw_windows(df):
    """(Xseq_raw, Xnext_raw, y_kw, ts): power channels in kW, exogenous MinMax-scaled.
    EXO_SEQ and EXO_NXT select the SAME 13 exogenous columns in the same order that
    exo_scaler was fit on, so a direct transform is correct for both."""
    seq = df[SEQ_FEATURES].to_numpy(np.float32).copy()
    nxt = df[NEXT_EXO].to_numpy(np.float32).copy()
    seq[:, EXO_SEQ] = exo_scaler.transform(seq[:, EXO_SEQ])
    nxt[:, EXO_NXT] = exo_scaler.transform(nxt[:, EXO_NXT])
    ykw = df["ev_kw"].to_numpy(np.float32)
    n = len(df) - LB
    Xs = np.zeros((n, LB, len(SEQ_FEATURES)), np.float32)
    Xn = np.zeros((n, len(NEXT_EXO)), np.float32)
    y = np.zeros(n, np.float32)
    for i in range(n):
        Xs[i] = seq[i:i+LB]; Xn[i] = nxt[i+LB]; y[i] = ykw[i+LB]
    ts = pd.to_datetime(df["timestamp"], utc=True).to_numpy()[LB:]
    return Xs, Xn, y, ts


# ----- split helpers (operate on already-built windows) -----
def split(Xs, Xn, y, ts):
    s = HOLD * 96
    tr = (Xs[:-s], Xn[:-s], y[:-s]); ho = (Xs[-s:], Xn[-s:], y[-s:])
    return tr, ho

def first_n(tr, n_days):
    s = min(n_days * 96, len(tr[2])); return tr[0][:s], tr[1][:s], tr[2][:s]


# ============================ REPRESENTATIONS ============================
# Each returns transform(Xs,Xn,y) -> (Xs',Xn',y', denorm) where denorm(pred)->kW.
# Global reps fit params on SOURCE power (training portion) once.

class GlobalRep:
    """Static, source-fit scalar transform f applied to all power channels + y."""
    def __init__(self, name, fit_fn, fwd, inv):
        self.name, self.fit_fn, self.fwd, self.inv = name, fit_fn, fwd, inv
        self.p = None
    def fit(self, src_power_train):
        self.p = self.fit_fn(src_power_train)
    def apply(self, Xs, Xn, y):
        Xs2, Xn2 = Xs.copy(), Xn.copy()
        Xs2[:, :, PW_SEQ] = self.fwd(Xs2[:, :, PW_SEQ], self.p)
        Xn2[:, PW_NXT] = self.fwd(Xn2[:, PW_NXT], self.p)
        y2 = self.fwd(y, self.p)
        denorm = lambda pred: self.inv(pred, self.p)
        return Xs2, Xn2, y2, denorm

class PerUnitRep:
    """DOMAIN-INFORMED: divide each SITE's power by its own capacity (per-unit)."""
    name = "per_unit"
    def __init__(self): self.base = None
    def fit_site(self, power_all): self.base = float(np.quantile(power_all, 0.999))
    def apply(self, Xs, Xn, y):
        b = self.base
        Xs2, Xn2 = Xs.copy(), Xn.copy()
        Xs2[:, :, PW_SEQ] /= b; Xn2[:, PW_NXT] /= b
        return Xs2, Xn2, y / b, (lambda pred: pred * b)

class RevINRep:
    """RevIN (no affine): per-window standardise power by lookback mean/std of ev_kw."""
    name = "revin"
    def apply(self, Xs, Xn, y):
        mu = Xs[:, :, EVKW_SEQ].mean(axis=1)                       # (N,)
        sg = Xs[:, :, EVKW_SEQ].std(axis=1) + 1e-6
        Xs2, Xn2 = Xs.copy(), Xn.copy()
        Xs2[:, :, PW_SEQ] = (Xs2[:, :, PW_SEQ] - mu[:, None, None]) / sg[:, None, None]
        Xn2[:, PW_NXT] = (Xn2[:, PW_NXT] - mu[:, None]) / sg[:, None]
        y2 = (y - mu) / sg
        denorm = lambda pred, _mu=mu, _sg=sg: pred * _sg + _mu     # per-window inverse
        return Xs2, Xn2, y2, denorm

class SANStyleRep:
    """OUR SAN-style reimplementation: per-slice normalisation of the lookback +
    a small linear predictor of the next-step local (mu,sigma) from the last slice's
    stats, fit on source. Denormalise the 1-step target with the predicted stats."""
    name = "san_style"
    def __init__(self): self.wmu = self.wsg = None
    def _slice_stats(self, Xs):
        L = Xs.shape[1]; p = L // SAN_SLICES
        ev = Xs[:, :, EVKW_SEQ]
        mus = np.stack([ev[:, s*p:(s+1)*p].mean(1) for s in range(SAN_SLICES)], 1)
        sgs = np.stack([ev[:, s*p:(s+1)*p].std(1) + 1e-6 for s in range(SAN_SLICES)], 1)
        return mus, sgs
    def fit_predictor(self, Xs, y):
        # target realised local stats ~ (y, |y-last_mu|); learn linear map from last slice
        mus, sgs = self._slice_stats(Xs)
        feat = np.c_[np.ones(len(Xs)), mus[:, -1], sgs[:, -1]]      # last-slice stats
        # ridge closed-form for mu-pred (target=y) and sigma-pred (target=last sigma)
        self.wmu = np.linalg.lstsq(feat, y, rcond=None)[0]
        self.wsg = np.linalg.lstsq(feat, sgs[:, -1], rcond=None)[0]
    def apply(self, Xs, Xn, y):
        mus, sgs = self._slice_stats(Xs)
        L = Xs.shape[1]; p = L // SAN_SLICES
        Xs2, Xn2 = Xs.copy(), Xn.copy()
        # per-slice normalise power channels of the lookback
        for s in range(SAN_SLICES):
            sl = slice(s*p, (s+1)*p)
            Xs2[:, sl][:, :, PW_SEQ] = (Xs2[:, sl][:, :, PW_SEQ]
                                        - mus[:, s][:, None, None]) / sgs[:, s][:, None, None]
        feat = np.c_[np.ones(len(Xs)), mus[:, -1], sgs[:, -1]]
        mu_pred = feat @ self.wmu
        sg_pred = np.abs(feat @ self.wsg) + 1e-6
        Xn2[:, PW_NXT] = (Xn2[:, PW_NXT] - mu_pred[:, None]) / sg_pred[:, None]
        y2 = (y - mu_pred) / sg_pred
        denorm = lambda pred, _m=mu_pred, _s=sg_pred: pred * _s + _m
        return Xs2, Xn2, y2, denorm


def make_reps():
    reps = [
        GlobalRep("minmax",
                  lambda p: dict(mx=float(np.quantile(p, 0.999)) + 1e-6),
                  lambda x, pp: x / pp["mx"], lambda x, pp: x * pp["mx"]),
        GlobalRep("zscore",
                  lambda p: dict(mu=float(p.mean()), sd=float(p.std()) + 1e-6),
                  lambda x, pp: (x - pp["mu"]) / pp["sd"], lambda x, pp: x * pp["sd"] + pp["mu"]),
        GlobalRep("robust",
                  lambda p: dict(md=float(np.median(p)),
                                 iqr=float(np.quantile(p, .75) - np.quantile(p, .25)) + 1e-6),
                  lambda x, pp: (x - pp["md"]) / pp["iqr"], lambda x, pp: x * pp["iqr"] + pp["md"]),
        GlobalRep("log",
                  lambda p: dict(lm=float(np.log1p(p).mean()), ls=float(np.log1p(p).std()) + 1e-6),
                  lambda x, pp: (np.log1p(np.clip(x, 0, None)) - pp["lm"]) / pp["ls"],
                  lambda x, pp: np.expm1(x * pp["ls"] + pp["lm"])),
        RevINRep(), SANStyleRep(), PerUnitRep(),
    ]
    return reps


def domain_distance(rep_src_y, rep_tgt_y):
    """1-Wasserstein between transformed source and target targets (subsampled)."""
    a = rep_src_y[np.random.default_rng(0).choice(len(rep_src_y), min(4000, len(rep_src_y)), False)]
    b = rep_tgt_y[np.random.default_rng(1).choice(len(rep_tgt_y), min(4000, len(rep_tgt_y)), False)]
    return float(wasserstein_distance(a, b))


SRC_DAYS = 150   # cap source-training span (memory); >>N_DAYS, ample for a source model


def main():
    # Build ONLY the windows we use, to keep host RAM bounded (full 672-lookback
    # window tensors are ~2.6 GB each otherwise). Capacities come straight from the
    # raw power columns (no windows needed).
    base_src = float(np.quantile(src.ev_kw.to_numpy(float), 0.999))
    base_tgt = float(np.quantile(tgt.ev_kw.to_numpy(float), 0.999))
    src_tail = src.iloc[-(SRC_DAYS*96 + LB):].reset_index(drop=True)
    sXs, sXn, sy, _ = raw_windows(src_tail)                   # source-train windows
    s_tr = (sXs, sXn, sy)
    src_pw_train = sy
    tgt_head = tgt.iloc[:LB + N_DAYS*96].reset_index(drop=True)
    hXs_r, hXn_r, hy_r, _ = raw_windows(tgt.iloc[-(LB + HOLD*96):].reset_index(drop=True))
    thXs, thXn, thy, _ = raw_windows(tgt_head)                # first N_DAYS target windows
    print(f"[abl] base_src={base_src:.1f} base_tgt={base_tgt:.1f} "
          f"src_win={len(sy)} tgt_head_win={len(thy)} holdout_win={len(hy_r)}", flush=True)

    # ---- incremental CSV + resume: skip (rep,cond) pairs already done ----
    if os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV)
        rows = prev.to_dict("records")
        done_cond = {(r["rep"], r["cond"]) for r in rows}
        done_reps = {r for r in prev.rep.unique()
                     if set(prev[prev.rep == r].cond) >= {"B0", "B3"}}
        print(f"[abl] resuming; done reps={sorted(done_reps)} "
              f"partial={sorted(done_cond - {(x,'B0') for x in done_reps} - {(x,'B3') for x in done_reps})}",
              flush=True)
    else:
        rows, done_cond, done_reps = [], set(), set()

    def flush_csv():
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    for rep in make_reps():
        if REP_SUBSET and rep.name not in REP_SUBSET:
            continue
        if rep.name in done_reps:
            continue
        keras.backend.clear_session()
        # ---- fit representation params on SOURCE ----
        if isinstance(rep, GlobalRep):
            rep.fit(src_pw_train)
        elif isinstance(rep, SANStyleRep):
            rep.fit_predictor(*(s_tr[0], s_tr[2]))
        # ---- transform SOURCE (train) ----
        if isinstance(rep, PerUnitRep):
            rep.base = base_src
        sXs_t, sXn_t, sy_t, s_denorm = rep.apply(*s_tr)
        # ---- transform TARGET: head (train) and holdout, from raw kW windows ----
        if isinstance(rep, PerUnitRep):
            rep.base = base_tgt                              # target uses its OWN capacity
        tr_t = rep.apply(thXs, thXn, thy)                    # first N_DAYS windows
        ho_t = rep.apply(hXs_r, hXn_r, hy_r)                 # holdout per-window stats
        t_tr = (tr_t[0], tr_t[1], tr_t[2])
        tXs_ho_t, tXn_ho_t, _, t_denorm_ho = ho_t
        ty_ho_kw = hy_r                                       # holdout truth stays in kW

        # domain distance in transformed target space (source-train vs target-head)
        dist = domain_distance(sy_t, tr_t[2])

        # ---- source model trained ONCE in this representation ----
        keras.utils.set_random_seed(0); srcm = build_ev_model(seed=0)
        srcm.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.Huber(0.5))
        srcm.fit([sXs_t, sXn_t], sy_t, epochs=12, batch_size=128, verbose=0)
        src_w = srcm.get_weights()

        # ---- transfer: B0 scratch vs B3 warm, K seeds, N=30 ----
        xn_tr = first_n(t_tr, N_DAYS)
        for cond in ("B0", "B3"):
            if (rep.name, cond) in done_cond:
                continue
            errs = []
            for seed in range(K_SEEDS):
                keras.backend.clear_session()
                keras.utils.set_random_seed(seed); m = build_ev_model(seed=seed)
                if cond == "B3": m.set_weights(src_w)
                m.compile(optimizer=keras.optimizers.Adam(1e-4), loss=keras.losses.Huber(0.5))
                m.fit([xn_tr[0], xn_tr[1]], xn_tr[2], epochs=10, batch_size=64, verbose=0)
                pred = m.predict([tXs_ho_t, tXn_ho_t], verbose=0).reshape(-1)
                pred_kw = np.clip(t_denorm_ho(pred), 0, None)
                rmse = float(np.sqrt(np.mean((pred_kw - ty_ho_kw) ** 2)))
                errs.append(rmse / (float(ty_ho_kw.mean()) + 1e-9))
            rows.append(dict(rep=rep.name, cond=cond, dist_W=round(dist, 4),
                             nrmse_mean=round(float(np.mean(errs)), 4),
                             nrmse_std=round(float(np.std(errs)), 4),
                             seeds=";".join(f"{e:.4f}" for e in errs)))
            print(rows[-1], flush=True)
            flush_csv()                                          # persist after every row

    R = pd.DataFrame(rows); R.to_csv(OUT_CSV, index=False)
    # ---- ranking on warm-start (B3) ----
    b3 = R[R.cond == "B3"].sort_values("nrmse_mean").reset_index(drop=True)
    print("\n=== EV warm-start (B3) ranking, N=30, {} seeds ===".format(K_SEEDS))
    print(b3[["rep", "nrmse_mean", "nrmse_std", "dist_W"]].to_string(index=False))
    # Wilcoxon: best vs each other (paired over seeds)
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
