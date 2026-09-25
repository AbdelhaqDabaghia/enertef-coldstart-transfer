"""
test_censoring.py -- the censored loss must be right about the one thing that
matters: a bound is not a target.

If the censored branch penalises a prediction for being BELOW the bound, the
method teaches the model the controller's floor instead of the site's demand,
which is the failure it exists to prevent. These tests pin that down, plus the
reconstruction algebra and the feature decontamination.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coldstart_transfer.censoring import (  # noqa: E402
    reconstruct_baseline, censored_loss, censored_loss_grad, decontaminate,
    decontaminated_reservoir, censoring_report)

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s  %s" % (name, detail))
        FAILURES.append(name)


def test_reconstruction():
    print("\nreconstruction de la demande de base")
    # slot 0: no control. 1: reduction absorbed. 2: reduction truncated.
    # 3: increase.
    y = np.array([40.0, 25.0, 0.0, 60.0])
    u = np.array([0.0, -15.0, -30.0, 20.0])
    y0, cens, bound = reconstruct_baseline(y, u)

    check("aucun controle -> y0 = y", y0[0] == 40.0, "got %.2f" % y0[0])
    check("reduction absorbee -> y0 = y - u", y0[1] == 40.0, "got %.2f" % y0[1])
    check("hausse -> y0 = y - u", y0[3] == 40.0, "got %.2f" % y0[3])
    check("seul le pas tronque est censure",
          list(cens) == [False, False, True, False], str(cens))
    check("la borne vaut -u", bound[2] == 30.0, "got %.2f" % bound[2])
    check("y0 censure porte la borne", y0[2] == 30.0, "got %.2f" % y0[2])

    r = censoring_report(y, u)
    check("rapport: 3 commandes, 1 censuree",
          r["commanded"] == 3 and r["censored"] == 1, str(r))
    check("energie cachee = borne x 0.25 h",
          abs(r["hidden_kwh"] - 7.5) < 1e-9, "%.3f" % r["hidden_kwh"])


def test_loss_treats_bound_as_bound():
    print("\nla perte traite une borne comme une borne")
    y0 = np.array([30.0])
    cens = np.array([True])
    bound = np.array([30.0])

    below = censored_loss(np.array([5.0]), y0, cens, bound)
    at = censored_loss(np.array([30.0]), y0, cens, bound)
    above = censored_loss(np.array([50.0]), y0, cens, bound)

    check("prediction SOUS la borne : cout nul", below == 0.0, "%.4f" % below)
    check("prediction A la borne : cout nul", at == 0.0, "%.4f" % at)
    check("prediction AU-DESSUS : cout positif", above > 0.0, "%.4f" % above)
    check("plus loin au-dessus coute plus cher",
          censored_loss(np.array([80.0]), y0, cens, bound) > above)

    # the decisive one: an ordinary squared loss would punish the model for
    # predicting 5 when the truth is only known to be <= 30
    naive = float((5.0 - 30.0) ** 2)
    check("une perte naive aurait puni 5 contre une borne de 30",
          naive > 0 and below == 0.0, "naive %.1f vs censored %.1f" % (naive, below))


def test_gradients():
    print("\ngradients")
    y0 = np.array([30.0, 40.0])
    cens = np.array([True, False])
    bound = np.array([30.0, np.inf])

    g = censored_loss_grad(np.array([10.0, 40.0]), y0, cens, bound)
    check("sous la borne : gradient nul", g[0] == 0.0, "%.4f" % g[0])
    check("residu nul non censure : gradient nul", g[1] == 0.0, "%.4f" % g[1])

    g = censored_loss_grad(np.array([50.0, 40.0]), y0, cens, bound)
    check("au-dessus de la borne : gradient positif (pousse vers le bas)",
          g[0] > 0.0, "%.4f" % g[0])

    g = censored_loss_grad(np.array([10.0, 30.0]), y0, cens, bound)
    check("sous-estimation non censuree : gradient negatif",
          g[1] < 0.0, "%.4f" % g[1])

    # numerical agreement with the analytic gradient
    p = np.array([45.0, 37.0])
    eps = 1e-6
    num = np.array([
        (censored_loss(p + eps * np.eye(2)[i], y0, cens, bound)
         - censored_loss(p - eps * np.eye(2)[i], y0, cens, bound)) / (2 * eps)
        for i in range(2)])
    ana = censored_loss_grad(p, y0, cens, bound)
    check("gradient analytique == numerique",
          np.allclose(num, ana, atol=1e-6), "num %s ana %s" % (num, ana))


def test_tobit_branch():
    print("\nbranche Tobit (sigma > 0)")
    y0 = np.array([30.0])
    cens = np.array([True])
    bound = np.array([30.0])
    far = censored_loss(np.array([5.0]), y0, cens, bound, sigma=5.0)
    near = censored_loss(np.array([29.0]), y0, cens, bound, sigma=5.0)
    over = censored_loss(np.array([50.0]), y0, cens, bound, sigma=5.0)
    check("Tobit: monotone en depassant la borne", far < near < over,
          "%.4f %.4f %.4f" % (far, near, over))
    check("Tobit: fini et positif partout",
          all(np.isfinite([far, near, over])) and far >= 0)


def test_decontamination():
    print("\ndecontamination des features")
    n = 300
    rng = np.random.default_rng(0)
    y0_true = 30.0 + 10.0 * np.sin(np.arange(n) / 12.0)
    u = np.zeros(n)
    u[100:150] = -40.0                       # a reduction that truncates
    y = np.maximum(y0_true + u, 0.0)

    f = decontaminate(y, u)
    check("lag_1 reconstruit depuis y0, pas y",
          abs(f["lag_1"][120] - f["_y0"][119]) < 1e-9)
    check("lag_1 depuis la serie observee serait faux",
          abs(f["lag_1"][120] - y[119]) > 1.0,
          "lag %.2f vs observe %.2f" % (f["lag_1"][120], y[119]))
    check("censored_frac nul avant tout controle", f["censored_frac"][50] == 0.0,
          "%.3f" % f["censored_frac"][50])
    check("censored_frac positif dans la fenetre tronquee",
          f["censored_frac"][140] > 0.0, "%.3f" % f["censored_frac"][140])
    check("censored_frac borne a 1",
          np.nanmax(f["censored_frac"]) <= 1.0 + 1e-12)
    check("moyennes glissantes causales (decalees de 1)",
          np.isnan(f["roll_1h_mean"][0]))


def test_reservoir_excludes_bounds():
    print("\ntampon de rejeu")
    y = np.array([40.0, 25.0, 0.0, 60.0])
    u = np.array([0.0, -15.0, -30.0, 20.0])
    rng = np.random.default_rng(0)
    buf, added = decontaminated_reservoir(None, y, u, size=10, rng=rng)
    check("les entrees censurees sont exclues", added == 3, "added %d" % added)
    check("aucune borne dans le tampon", 30.0 not in list(buf), str(buf))
    check("le tampon contient la base reconstruite",
          all(abs(v - 40.0) < 1e-9 for v in buf), str(buf))


if __name__ == "__main__":
    for t in (test_reconstruction, test_loss_treats_bound_as_bound, test_gradients,
              test_tobit_branch, test_decontamination, test_reservoir_excludes_bounds):
        t()
    print("\n%s" % ("tous les tests passent" if not FAILURES
                    else "%d ECHEC(S): %s" % (len(FAILURES), ", ".join(FAILURES))))
    sys.exit(1 if FAILURES else 0)
