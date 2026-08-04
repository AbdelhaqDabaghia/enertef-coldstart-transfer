# Cold-Start Continual-Learning Transfer Protocol — EV Demand (Luxembourg → UK)

Can a low-data target site bootstrap its EV demand forecaster by transferring
weights and/or Fisher importance from a mature Luxembourg production model, then
keep adapting without forgetting the source domain? Tested on a **real, published
independent target** (UK *Electric Nation* smart-charging trial), not a synthetic
split.

## TL;DR — findings

| Research question | Result |
|---|---|
| **RQ1** cold-start transfer benefit | **Warm-start helps a lot** — ~40% lower nRMSE than from-scratch at every N (1–90 days), 3 seeds, well separated. |
| **RQ2** stability–plasticity | **Replay wins the tradeoff** (best source-retention at modest target cost). EWC adds nothing over plain warm-start. |
| **RQ3** Fisher transferability | With Fisher **scales equalized** (source Fisher is ~211× the target's), transferred-vs-target Fisher are **comparable** — the earlier "transferred Fisher is harmful" was a scale artifact. The honest conclusion: **the Fisher's origin doesn't matter; EWC itself gives no benefit for this transfer.** |
| **EWC-variants bench** | True empirical Fisher, per-layer normalization, and an EWC+replay hybrid — **none beats warm-start (plasticity) or replay (retention)** over 3 seeds. |

**Headline (honest, defensible):** *for cold-start EV demand transfer, a simple
warm-start + replay beats every EWC variant tested; the sophisticated regulariser
does not help.* The domain shift is dominated by the demand distribution itself
(6× scale, Wasserstein 0.62) — weather transfers well (MMD² 0.034).

## Modules

| File | Role |
|---|---|
| `features.py` | Reconstruct the 21 SEQ_FEATURES; **hard** validation against the frozen production scalers. |
| `model.py` | `build_ev_model()` — exact replica of the 2026-07-18 production CNN-LSTM (fidelity verified, max\|Δ\|=0). |
| `windowing.py` | Flat frame → `(672,21)`/`(20,)`/scalar triples, frozen-scaler scaling, holdout/N-day splits. |
| `ewc.py` | Output-sensitivity Fisher proxy **and** true empirical Fisher; EWC penalty; Huber. |
| `trainer.py` | One parameterised loop for **B0–B5** + variants (Fisher source/target/empirical, normalization, replay). |
| `logger.py` | One CSV row per run (baseline, N, λ, β, target & source-retention nRMSE, wall-clock, seed). |
| `driver.py` | RQ1 grid (B0/B3/B4 × N × seeds). |
| `driver_source.py` | RQ2/RQ3 (B1/B4/B5 + source-retention). |
| `driver_fisher_control.py` | RQ3 fairness control (normalized Fisher + λ sweep). |
| `driver_ewc_bench.py` | Honest improved-EWC bench (V1/V2/V3 vs B3/B5). |
| `analysis.py` | The four figures. |

Baselines: **B0** from-scratch · **B1** warm+EWC(source Fisher) · **B2/B3** warm,
no-EWC · **B4** warm+EWC(target Fisher) · **B5** warm+replay.

## Reproduce

```bash
# 1. source features (EV from local ECC_master, weather re-fetched from Open-Meteo)
python scripts/build_lux_source.py            # -> Data/lux_source_features.csv (scaler-validated)

# 2. sanity check (reload prod model + scalers, non-degenerate MAE)
python scripts/sanity_source.py

# 3. experiments (GPU via WSL2; drop the wsl wrapper to run CPU on Windows)
wsl -d Ubuntu-24.04 -u root -- bash scripts/wsl_run.sh -m coldstart_transfer.driver          # RQ1
wsl ... wsl_run.sh -m coldstart_transfer.driver_source          # RQ2/RQ3
wsl ... wsl_run.sh -m coldstart_transfer.driver_fisher_control  # RQ3 control
wsl ... wsl_run.sh -m coldstart_transfer.driver_ewc_bench       # EWC bench

# 4. domain distance + figures
python scripts/domain_distance.py
python -m coldstart_transfer.analysis          # -> Data/results/figures/*.png
```

Deps: `requirements-coldstart.txt`. GPU: `tensorflow[and-cuda]` in WSL2 (verified
on an RTX 5070 / CUDA 12.9); native-Windows TF is CPU-only.

## Simplifications & limitations (stated, not hidden)

- **Single target site, single architecture, single task** → the "EWC doesn't
  help" claim is empirical for *this* transfer; it cannot be generalised. Relating
  domain distance to the transfer gap (RQ3's aim) needs ≥2–3 target sites.
- **UK weather = single point** (Bristol, Open-Meteo) for the whole trial service
  area; **source weather re-fetched** from Open-Meteo (min/max reproduce the scaler
  exactly, but individual values are an approximation of the original DB snapshot).
- **UK data reflects a smart-charging trial intervention**, not "natural" demand.
- **Absolute nRMSE is high (~0.52)** even warm-started — the large demand shift
  limits transfer quality; the benefit is *relative* to from-scratch.
- **3 seeds** (run-to-run CV up to ~35% observed) — enough to separate the main
  effects here, but 5–10 recommended before strong claims.
- Production model is fine-tuned beyond the scaler-freeze state, so the sanity MAE
  (~18.6 kW) is reasonable/non-degenerate rather than an exact reproduction of the
  13.58 kW gate.
