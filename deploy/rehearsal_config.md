# Switching nightly adaptation from the regulariser to bounded rehearsal

**No code change is required.** `enertef-cl-trainer/sagemaker-training/train_ev_cnn_lstm.py`
already implements bounded rehearsal with a reservoir-sampled buffer. It is
dormant because the environment disables it. The switch is three variables.

## The change

| variable | now | set to | effect |
|---|---|---|---|
| `LAMBDA_EWC` | `1000.0` | `0.0` | disables the output-sensitivity penalty |
| `REPLAY_BUFFER_SIZE` | `0` | `2000` | enables the reservoir, at the capacity used in our experiments |
| `REPLAY_BUFFER_KEY` | *(empty)* | `<EXPERIMENT_PREFIX>/replay_buffer.npz` | chains the buffer across cycles |

Leave `DER_ALPHA` at `0.0`. DER++ landed between OSR and replay in our runs
without separating from either, so it adds a moving part for no measured gain.

## Why these are exactly the right knobs

`ewc.py:134` returns `(lambda_ewc / 2.0) * penalty`, so `LAMBDA_EWC=0.0` is an
exact no-op rather than an approximate one — the penalty term contributes
nothing to the gradient.

`data_loader.py:373 mix_buffer_into_training` concatenates the buffer with the
current window **before** `scale_data`, so the one frozen scaler is applied
uniformly to both. This ordering matters and is already correct; mixing after
scaling would apply the transform inconsistently.

`data_loader.py:302 reservoir_update` maintains the bounded buffer with
reservoir sampling against `n_seen`, which is what makes the buffer a uniform
sample of everything seen rather than a recency window.

`download_replay_buffer` treats a missing key as an empty reservoir rather than
an error, so cycle 0 needs no special handling.

## What the evidence is, and what it is not

Two independent lines favour rehearsal:

- **Cross-site transfer, causal features, 10 seeds** (experiment e13). Bounded
  rehearsal beats plain fine-tuning by 0.346 nRMSE, unanimously, d_z = −17.57;
  the output-sensitivity regulariser is statistically indistinguishable from
  plain fine-tuning (+0.0008, p = 0.70). Rehearsal is the only arm that beats
  naive persistence.
- **Sequential 12-cycle deployment, 5 seeds** (`m1_champion_challenger.csv`).
  Replay holds mid-stream backward transfer at −1.39 kW against the
  regulariser's +9.68.

**Be clear about the gap.** The first is a *transfer* setting — adapt on one
site, measure retention on another. Production does something different: it
adapts nightly on recent data from the same site. The second is the right
setting but was measured on the leaky feature pipeline.

So: no experiment has yet measured rehearsal against the regulariser *in the
production adaptation setting under causal features*. Two lines of evidence
point the same way and the mechanism is plausible — rehearsal is the only
mechanism that keeps seeing the distribution it is asked to retain — but this
is a decision taken on converging indirect evidence, not on a direct
measurement. Re-running the 12-cycle study causally would close it.

## Rollout

1. **Do not change this and the feature convention in the same cycle.** The
   causal retrain (`ev_cnn_lstm_causal_full_warm_e20_s1.keras`) is already a
   large change to what the model sees. Ship it, let a few cycles settle, then
   switch the adaptation mechanism. Two simultaneous changes to a nightly job
   are indistinguishable in the logs when something moves.
2. Set the three variables above.
3. Watch `n_seen` and buffer size in the `[REPLAY]` log lines for the first few
   cycles: size should climb to 2000 and then hold while `n_seen` keeps rising.
   A buffer stuck below capacity means the key is wrong and each cycle is
   starting from an empty reservoir — which silently reduces this to plain
   fine-tuning.
4. The promotion gate should be `deploy/validation_v2.py` before this change,
   not after. Rehearsal trades plasticity for retention (recent-window MAE rose
   from 8.5 to 13.8 kW in the sequential study), so a gate with an absolute
   floor matters more once it is on, not less.

## Cost

One extra concatenation per cycle and up to 2000 additional training samples —
negligible against a 90-day window. The buffer file is a few MB in S3.
