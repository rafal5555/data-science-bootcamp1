"""Walidacja krocząca (walk-forward), metryki i test Diebolda–Mariano."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from scipy import stats


def walk_forward(prices: pd.Series, models: list, h: int = 5, test_size: int = 500,
                 refit_every: int = 21) -> pd.DataFrame:
    """Prognozy poza próbą. Dla bloku punktów startowych o ∈ [s, s+refit_every) modele są uczone
    wyłącznie na logp[:s+1] (z celami o+h <= s), więc żadna informacja z przyszłości nie przecieka."""
    logp = np.log(prices.to_numpy(dtype=float))
    n = len(logp)
    last = n - 1 - h  # ostatni punkt, dla którego znamy realizację
    first = last - test_size + 1
    need = max(getattr(m, "min_history", 1) for m in models) + h + 250
    if first < need:
        raise ValueError(f"Za krótki szereg: {n} obs.; potrzeba ok. {need + test_size + h}. "
                         "Zmniejsz --test-size lub weź dłuższą historię.")
    models = [copy.deepcopy(m) for m in models]
    origins = np.arange(first, last + 1)
    preds = {m.name: np.empty(len(origins)) for m in models}
    for b in range(0, len(origins), refit_every):
        block = origins[b:b + refit_every]
        train = logp[: block[0] + 1]
        for m in models:
            m.fit(train, h)
            preds[m.name][b:b + len(block)] = m.predict(logp, block)
    out = pd.DataFrame(preds, index=prices.index[origins])
    out.insert(0, "actual", logp[origins + h] - logp[origins])
    return out


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int) -> tuple[float, float]:
    """Test DM (Diebold & Mariano, 1995, JBES) z poprawką małej próby Harveya, Leybourne'a
    i Newbolda (1997, IJF). Strata kwadratowa. H0: równa trafność. Statystyka < 0 → model 1 lepszy.
    Wariancja HAC z h-1 opóźnieniami, bo cele h-dniowe nakładają się."""
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dc = d - d.mean()
    gamma = [dc @ dc / n] + [dc[k:] @ dc[:-k] / n for k in range(1, h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return np.nan, np.nan
    dm = d.mean() / np.sqrt(var)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(dm), df=n - 1)
    return float(dm), float(p)


def score(results: pd.DataFrame, h: int, benchmark: str = "RandomWalk") -> pd.DataFrame:
    """Metryki poza próbą dla każdego modelu względem benchmarku."""
    y = results["actual"].to_numpy()
    e_bench = y - results[benchmark].to_numpy()
    rows = []
    for name in results.columns.drop("actual"):
        p = results[name].to_numpy()
        e = y - p
        nz = p != 0
        dm, pval = diebold_mariano(e, e_bench, h) if name != benchmark else (np.nan, np.nan)
        rows.append({
            "model": name,
            "RMSE": np.sqrt(np.mean(e ** 2)),
            "MAE": np.mean(np.abs(e)),
            # R²oos (Campbell & Thompson, 2008, RFS) liczony względem benchmarku
            "R2oos_vs_RW": 1 - (e @ e) / (e_bench @ e_bench),
            # trafność kierunku; dla prognoz zerowych nieokreślona
            "hit_rate": np.mean(np.sign(p[nz]) == np.sign(y[nz])) if nz.any() else np.nan,
            "DM_stat": dm,
            "DM_p": pval,
        })
    return pd.DataFrame(rows).set_index("model")
