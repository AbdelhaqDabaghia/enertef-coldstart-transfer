# Cold-Start Continual-Learning Transfer for EV & PV Forecasting

Reproducible pipeline for the paper *"When Simple Beats Sophisticated: A
Cold-Start Continual-Learning Transfer Protocol"* — two real cross-country
transfers (EV demand: Luxembourg → UK Electric Nation; PV generation:
Luxembourg → Konstanz/OPSD).

## Layout
- `coldstart_transfer/` — the package: `model.py` / `pv.py` (verified model
  builders), `windowing.py`, `ewc.py` (Fisher proxy + empirical), `trainer.py`
  (parameterised B0–B5), `logger.py`, drivers (`driver*.py`), `analysis.py`
  (the 6 figures), `README.md` (findings + limitations).
- `scripts/` — data builders (`build_lux_source.py`, `build_pv_source.py`,
  `build_pv_target.py`), sanity/domain-distance, GPU runner (`wsl_run.sh`).
- `Data/results/` — per-run CSV logs (real 3-seed GPU runs, final) + figures.
- `Data/models/` — the frozen production EV/PV models + scalers.
- `Data/` — feature CSVs (EV UK, PV Konstanz, LU sources) + raw ECC_master.
- `paper/` — IEEE LaTeX source + figures (compiles on Overleaf).

## Regenerate the figures
```
python -m coldstart_transfer.analysis      # writes Data/results/figures/*.png
```
See `figures_repro/README_figures.md`-style mapping in `coldstart_transfer/README.md`.

## Run experiments (GPU via WSL2, or CPU on Windows)
```
wsl -d Ubuntu-24.04 -u root -- bash scripts/wsl_run.sh -m coldstart_transfer.driver          # EV RQ1
wsl ... wsl_run.sh -m coldstart_transfer.driver_pv                                            # PV RQ1
```
Deps: `requirements-coldstart.txt`. GPU: `tensorflow[and-cuda]` in WSL2.

## Note on credentials
No secrets are committed. DB/S3 access reads credentials from environment
variables only.
