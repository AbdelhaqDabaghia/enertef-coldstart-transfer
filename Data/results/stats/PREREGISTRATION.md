# Equivalence-margin pre-declaration (TOST)

**Declared 2026-09-04, before running the 10-seed campaign** — i.e. before any
10-seed result was observed. This fixes the equivalence margins δ *a priori*, on
domain grounds, which is what licenses the TOST equivalence conclusions in the
revised paper. The margins were NOT read off the data.

| metric                    | margin δ | rationale |
|---------------------------|:--------:|-----------|
| `target_nrmse` (plasticity) | **0.02** | 2 percentage-points of relative error — below the smallest gap that would change an operational cold-start decision at the 15-min horizon. |
| `source_retention_nrmse` (stability) | **0.05** | 5 pp of relative error on the *source* task; a wider band is defensible because source retention is a secondary objective, not the deployment target. |

Two one-sided tests at α = 0.05 ⇒ equivalence iff the 90% CI of the seed-matched
mean difference lies entirely within [−δ, +δ]. Applied vs. the warm-start
reference (`B3` / `B3_warm`) for the EWC-variant, RQ2/RQ3, and full CL-suite
benchmarks. See `coldstart_transfer/stats_report.py`.
