"""Modele prognozujące h-dniowy log-zwrot: y(o) = log P[o+h] - log P[o].

Wspólny interfejs:
    fit(logp, h)          – uczy model na log-cenach logp[0..n-1]; używa tylko par (o, o+h) z o+h <= n-1.
    predict(logp, origins) – prognoza dla każdego punktu o, korzystając wyłącznie z logp[:o+1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _targets(logp: np.ndarray, origins: np.ndarray, h: int) -> np.ndarray:
    return logp[origins + h] - logp[origins]


class RandomWalk:
    """Błądzenie losowe (prognoza zwrotu = 0). Benchmark, którego w kursach walut
    modele strukturalne zwykle nie pokonują (Meese & Rogoff, 1983, J. Int. Econ.)."""

    name = "RandomWalk"
    min_history = 1

    def fit(self, logp, h):
        return self

    def predict(self, logp, origins):
        return np.zeros(len(origins))


class HistoricalMean:
    """Średni historyczny h-dniowy zwrot z okresu uczącego (dryf)."""

    name = "HistMean"
    min_history = 2

    def fit(self, logp, h):
        self.mu_ = (logp[-1] - logp[0]) / (len(logp) - 1) * h
        return self

    def predict(self, logp, origins):
        return np.full(len(origins), self.mu_)


class DLinear:
    """DLinear (Zeng i in., AAAI 2023, arXiv:2205.13504).

    Okno wejściowe jest rozkładane średnią ruchomą na trend i resztę; na każdej składowej
    działa jedna warstwa liniowa, prognoza jest bezpośrednia (direct multi-step).
    Wejście normalizujemy odjęciem ostatniej wartości (wariant „NLinear” z tej samej pracy),
    co dla log-cen oznacza pracę na skumulowanych zwrotach i usuwa niestacjonarność poziomu.
    Zamiast Adama/MSE wagi wyznaczamy w formie zamkniętej (regresja grzbietowa) — to ten sam
    model liniowy z karą L2, deterministyczny i szybki przy częstym ponownym uczeniu.
    """

    name = "DLinear"

    def __init__(self, window: int = 60, kernel: int = 25, alpha: float = 0.1):
        if kernel % 2 == 0:
            raise ValueError("kernel musi być nieparzysty")
        self.window, self.kernel, self.alpha = window, kernel, alpha
        self.min_history = window

    def _features(self, logp, origins):
        win = sliding_window_view(logp, self.window)[origins - self.window + 1]
        x = win - win[:, -1:]
        pad = self.kernel // 2  # jak w oryginale: dopełnienie skrajnymi wartościami
        xp = np.concatenate([np.repeat(x[:, :1], pad, 1), x, np.repeat(x[:, -1:], pad, 1)], axis=1)
        csum = np.cumsum(np.pad(xp, ((0, 0), (1, 0))), axis=1)
        trend = (csum[:, self.kernel:] - csum[:, :-self.kernel]) / self.kernel
        return np.hstack([trend, x - trend])

    def fit(self, logp, h):
        self.h = h
        origins = np.arange(self.window - 1, len(logp) - h)
        X, y = self._features(logp, origins), _targets(logp, origins, h)
        # standaryzacja cech, kara L2 = alpha na obserwację (alpha jest niezależne od długości próby)
        Xm, Xs, ym = X.mean(0), X.std(0) + 1e-12, y.mean()
        Z = (X - Xm) / Xs
        w = np.linalg.solve(Z.T @ Z + self.alpha * len(y) * np.eye(Z.shape[1]), Z.T @ (y - ym))
        self.w_ = w / Xs
        self.b_ = ym - Xm @ self.w_
        return self

    def predict(self, logp, origins):
        return self._features(logp, np.asarray(origins)) @ self.w_ + self.b_


def make_features(logp: np.ndarray) -> pd.DataFrame:
    """Przyczynowe (wyłącznie z przeszłości) cechy techniczne: momentum, zmienność, odległość od średnich."""
    s = pd.Series(logp)
    r = s.diff()
    f = pd.DataFrame({f"mom_{k}": s - s.shift(k) for k in (1, 5, 20, 60, 120)})
    f["vol_20"] = r.rolling(20).std()
    f["vol_60"] = r.rolling(60).std()
    f["vol_ratio"] = f["vol_20"] / f["vol_60"]
    f["ma_50"] = s - s.rolling(50).mean()
    f["ma_200"] = s - s.rolling(200).mean()
    f["mom_20_scaled"] = f["mom_20"] / (f["vol_20"] * np.sqrt(20))  # momentum skorygowane o ryzyko
    return f


class GradientBoosting:
    """Drzewa wzmacniane gradientowo na cechach technicznych. Metody drzewiaste należały do
    najlepszych w porównaniu ML dla premii za ryzyko akcji (Gu, Kelly & Xiu, 2020, RFS).
    Silna regularyzacja, bo stosunek sygnału do szumu w zwrotach jest bardzo niski."""

    name = "GBoost"
    min_history = 200

    def __init__(self, max_iter: int = 150, learning_rate: float = 0.03, max_depth: int = 3,
                 min_samples_leaf: int = 100, seed: int = 0):
        self.params = dict(max_iter=max_iter, learning_rate=learning_rate, max_depth=max_depth,
                           min_samples_leaf=min_samples_leaf, l2_regularization=1.0, random_state=seed)

    def fit(self, logp, h):
        from sklearn.ensemble import HistGradientBoostingRegressor

        origins = np.arange(self.min_history - 1, len(logp) - h)
        X = make_features(logp).to_numpy()[origins]
        self.model_ = HistGradientBoostingRegressor(**self.params).fit(X, _targets(logp, origins, h))
        return self

    def predict(self, logp, origins):
        origins = np.asarray(origins)
        # cechy liczone tylko z logp[:max(origins)+1] — gwarancja braku podglądania przyszłości
        X = make_features(logp[: origins.max() + 1]).to_numpy()[origins]
        return self.model_.predict(X)


class Chronos:
    """Chronos-Bolt — prekonfigurowany model bazowy szeregów czasowych, użyty zero-shot
    (Ansari i in., TMLR 2024, arXiv:2403.07815). Opcjonalny: wymaga `pip install chronos-forecasting`
    i pobrania wag z Hugging Face. Prognozą jest mediana rozkładu ceny w horyzoncie h."""

    name = "Chronos"

    def __init__(self, model_id: str = "amazon/chronos-bolt-small", context: int = 512, batch: int = 64):
        self.model_id, self.context, self.batch = model_id, context, batch
        self.min_history = 64
        self.pipe_ = None

    def fit(self, logp, h):
        self.h = h
        if self.pipe_ is None:
            from chronos import BaseChronosPipeline

            self.pipe_ = BaseChronosPipeline.from_pretrained(self.model_id, device_map="cpu")
        return self

    def predict(self, logp, origins):
        import torch

        price = np.exp(logp)
        out = []
        for i in range(0, len(origins), self.batch):
            chunk = origins[i:i + self.batch]
            ctx = [torch.tensor(price[max(0, o - self.context + 1): o + 1]) for o in chunk]
            q, _ = self.pipe_.predict_quantiles(ctx, prediction_length=self.h, quantile_levels=[0.5])
            med = q[:, self.h - 1, 0].numpy()
            out.append(np.log(med) - logp[chunk])
        return np.concatenate(out)


def default_models(with_chronos: bool = False) -> list:
    models = [RandomWalk(), HistoricalMean(), DLinear(), GradientBoosting()]
    if with_chronos:
        models.append(Chronos())
    return models
