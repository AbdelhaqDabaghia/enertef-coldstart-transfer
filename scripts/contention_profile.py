"""
contention_profile.py -- PER-LAYER contention profiling for the CPA idea.

mechanism_b_fisher_gradient.py established GLOBALLY that the source-Fisher-
important parameters are also the ones the target needs to move (cosine .82,
62% of target-gradient mass in the top-10% Fisher params). CPA needs the same
question answered PER LAYER: is a given layer *contended* (source depends on it
AND target must change it), and can source and target share it?

Two distinct quantities, deliberately kept separate:

  1. CONTENTION (unsigned).  F and |g_T| are both non-negative, so their cosine
     lives in [0,1] and is upward-biased in high dimension -- it cannot measure
     directional opposition, only importance overlap. We therefore report:
       - enrich10: fraction of |g_T| mass sitting in the top-10% Fisher params
         of that layer, divided by 0.10.  Expected 1.0 under independence,
         so it is self-normalising and directly interpretable.
       - cos_F_absG plus a PERMUTATION NULL (shuffle F within the layer): the
         z-score says whether the raw cosine is higher than chance for a
         non-negative vector pair of that dimension.

  2. DIRECTIONAL AGREEMENT (signed).  cos(g_S, g_T) on the raw signed gradients.
     This is the quantity that says whether source and target pull the same way,
     i.e. whether a contended layer can still be SHARED rather than duplicated.

Allocation rule under test (2x2): low contention -> freeze; high contention +
aligned -> share; high contention + opposed -> duplicate.

Writes Data/results/contention_profile.csv (one row per layer per pair) and
prints a per-pair summary. Read-only w.r.t. every existing result file.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from coldstart_transfer import ewc as ewc_mod

M_POINTS = int(os.environ.get("M_POINTS", "500"))
N_DAYS = int(os.environ.get("N_DAYS", "30"))
SEED = int(os.environ.get("SEED", "0"))
N_PERM = int(os.environ.get("N_PERM", "200"))
OUT = os.environ.get("OUT", "Data/results/contention_profile.csv")


def _flat(t):
    return np.asarray(t).ravel()


def signed_grad(model, X_seq, X_next, y, idx):
    """Signed gradient of the Huber loss w.r.t. every trainable variable."""
    tvars = model.trainable_variables
    with tf.GradientTape() as tape:
        yhat = model([tf.constant(X_seq[idx]), tf.constant(X_next[idx])],
                     training=False)
        loss = ewc_mod.huber(tf.constant(y[idx]), yhat)
    grads = tape.gradient(loss, tvars)
    return [np.zeros_like(v.numpy()) if g is None else g.numpy()
            for g, v in zip(grads, tvars)]


def enrichment(f, g_abs, q=0.90):
    """Fraction of |g| mass in the top-(1-q) Fisher params, / (1-q).

    1.0 == chance. Returns NaN for degenerate layers (all-zero F or g)."""
    if f.size < 10 or g_abs.sum() <= 0 or f.sum() <= 0:
        return np.nan
    thr = np.quantile(f, q)
    sel = f >= thr
    if sel.sum() == 0:
        return np.nan
    share = g_abs[sel].sum() / g_abs.sum()
    return float(share / (sel.sum() / f.size))


def cos_nonneg(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def perm_null_z(f, g_abs, n_perm, rng):
    """z-score of cos(F,|g|) against shuffling F within the layer.

    Controls the upward bias of a cosine between two non-negative vectors."""
    obs = cos_nonneg(f, g_abs)
    if not np.isfinite(obs) or f.size < 10:
        return obs, np.nan, np.nan
    null = np.empty(n_perm)
    fp = f.copy()
    for i in range(n_perm):
        rng.shuffle(fp)
        null[i] = cos_nonneg(fp, g_abs)
    mu, sd = float(null.mean()), float(null.std())
    z = (obs - mu) / sd if sd > 0 else np.nan
    return obs, mu, z


def profile_pair(pair, model, F, gT, gS, rng):
    """One row per trainable variable, plus a POOLED row over all of them."""
    tvars = model.trainable_variables
    rows = []
    for i, v in enumerate(tvars):
        f = _flat(F[i].numpy() if hasattr(F[i], "numpy") else F[i])
        gt = _flat(gT[i])
        gs = _flat(gS[i])
        gt_abs = np.abs(gt)
        obs, null_mu, z = perm_null_z(f, gt_abs, N_PERM, rng)
        rows.append(dict(
            pair=pair, layer=v.name, idx=i, n_params=f.size,
            enrich10=enrichment(f, gt_abs),
            cos_F_absG=obs, cos_F_absG_null=null_mu, cos_F_absG_z=z,
            cos_gS_gT=cos_nonneg(gs, gt),          # signed: alignment
            fisher_mass=float(f.sum()),
            gT_mass=float(gt_abs.sum()),
        ))
    # pooled over the whole network, to reproduce the mechanism-b numbers
    f = np.concatenate([_flat(x.numpy() if hasattr(x, "numpy") else x) for x in F])
    gt = np.concatenate([_flat(x) for x in gT])
    gs = np.concatenate([_flat(x) for x in gS])
    gt_abs = np.abs(gt)
    fn = f / (f.sum() + 1e-12)
    gn = gt_abs / (gt_abs.sum() + 1e-12)
    obs, null_mu, z = perm_null_z(fn, gn, N_PERM, rng)
    rows.append(dict(
        pair=pair, layer="__POOLED__", idx=-1, n_params=f.size,
        enrich10=enrichment(fn, gn),
        cos_F_absG=obs, cos_F_absG_null=null_mu, cos_F_absG_z=z,
        cos_gS_gT=cos_nonneg(gs, gt),
        fisher_mass=float(f.sum()), gT_mass=float(gt_abs.sum()),
    ))
    return rows


def pair_ev():
    """Luxembourg EV source model -> UK Electric Nation target."""
    from coldstart_transfer.model import build_ev_model, load_production_weights
    from coldstart_transfer.windowing import make_windows, split_holdout, first_n_days

    sc = joblib.load("Data/models/ev_scalers.joblib")
    m = build_ev_model()
    load_production_weights(m, "Data/models/ev_cnn_lstm_20260718.keras")

    # --- source side: same slice mechanism_b uses, for comparability
    src = pd.read_csv("Data/lux_source_features.csv")
    tail = (60 + 7) * 96 + 672
    sX, sN, sy, sts = make_windows(src.iloc[-tail:].reset_index(drop=True), sc)
    spool, _ = split_holdout(sX, sN, sy, sts, 7)
    F = ewc_mod.estimate_fisher(m, spool[0], spool[1],
                                m_points=M_POINTS, batch_size=64)
    rng = np.random.default_rng(SEED)
    sidx = rng.choice(len(spool[2]), min(M_POINTS, len(spool[2])), replace=False)
    gS = signed_grad(m, spool[0], spool[1], spool[2], sidx)

    # --- target side: first N_DAYS of the target stream, holdout excluded
    tgt = pd.read_csv("Data/uk_ev_features_full.csv")
    tX, tN, ty, tts = make_windows(tgt, sc)
    tr, _ = split_holdout(tX, tN, ty, tts, 14)
    xs, xn, yy, _ = first_n_days(*tr, n_days=N_DAYS)
    tidx = rng.choice(len(yy), min(M_POINTS, len(yy)), replace=False)
    gT = signed_grad(m, xs, xn, yy, tidx)
    return "EV_LU2UK", m, F, gT, gS


def pair_pv():
    """Luxembourg PV source model -> Konstanz target."""
    from coldstart_transfer.pv import build_pv_model, load_pv_production_weights, make_pv_windows
    from coldstart_transfer.windowing import split_holdout, first_n_days

    sc = joblib.load("Data/models/pv_scalers.joblib")
    m = build_pv_model()
    load_pv_production_weights(m, "Data/models/pv_lstm_openmeteo.keras")

    src = pd.read_csv("Data/pv_target/lux_pv_source_features.csv")
    sX, sN, sy, sts = make_pv_windows(src, sc)
    spool, _ = split_holdout(sX, sN, sy, sts, 7)
    F = ewc_mod.estimate_fisher(m, spool[0], spool[1],
                                m_points=M_POINTS, batch_size=64)
    rng = np.random.default_rng(SEED)
    sidx = rng.choice(len(spool[2]), min(M_POINTS, len(spool[2])), replace=False)
    gS = signed_grad(m, spool[0], spool[1], spool[2], sidx)

    tgt = pd.read_csv("Data/pv_target/konstanz_pv_features.csv")
    tX, tN, ty, tts = make_pv_windows(tgt, sc)
    tr, _ = split_holdout(tX, tN, ty, tts, 14)
    xs, xn, yy, _ = first_n_days(*tr, n_days=N_DAYS)
    tidx = rng.choice(len(yy), min(M_POINTS, len(yy)), replace=False)
    gT = signed_grad(m, xs, xn, yy, tidx)
    return "PV_LU2KN", m, F, gT, gS


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for builder in (pair_ev, pair_pv):
        try:
            name, m, F, gT, gS = builder()
        except Exception as e:                       # missing data -> skip, say so
            print(f"[contention] SKIP {builder.__name__}: {type(e).__name__}: {e}")
            continue
        r = profile_pair(name, m, F, gT, gS, rng)
        rows += r
        pooled = [x for x in r if x["layer"] == "__POOLED__"][0]
        per = [x for x in r if x["layer"] != "__POOLED__"
               and np.isfinite(x["enrich10"])]
        e = np.array([x["enrich10"] for x in per])
        a = np.array([x["cos_gS_gT"] for x in per])
        print(f"\n[{name}] pooled cos(F,|gT|)={pooled['cos_F_absG']:.3f} "
              f"(null {pooled['cos_F_absG_null']:.3f}, z={pooled['cos_F_absG_z']:.1f})  "
              f"enrich10={pooled['enrich10']:.2f}x  cos(gS,gT)={pooled['cos_gS_gT']:+.3f}")
        print(f"[{name}] per-layer enrich10: min={e.min():.2f} med={np.median(e):.2f} "
              f"max={e.max():.2f} spread={e.max()-e.min():.2f}  (n_layers={len(per)})")
        print(f"[{name}] per-layer cos(gS,gT): min={a.min():+.3f} med={np.median(a):+.3f} "
              f"max={a.max():+.3f}  negative in {(a < 0).sum()}/{len(a)} layers")

    if not rows:
        raise SystemExit("[contention] no pair could be profiled")
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\n[contention] wrote {OUT}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
