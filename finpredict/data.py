"""Ładowanie szeregów cenowych: Yahoo Finance, plik CSV lub dane syntetyczne."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Gotowe zestawy tickerów w notacji Yahoo Finance.
PRESETS: dict[str, list[str]] = {
    "waluty": ["EURUSD=X", "EURPLN=X", "USDPLN=X", "GBPUSD=X", "USDJPY=X"],
    "kruszce": ["GC=F", "SI=F", "PL=F", "PA=F", "HG=F"],  # złoto, srebro, platyna, pallad, miedź
    "spolki": ["AAPL", "MSFT", "NVDA", "PKO.WA", "PKN.WA"],
}


def load_yahoo(ticker: str, start: str = "2010-01-01", end: str | None = None) -> pd.Series:
    """Pobiera skorygowane ceny zamknięcia z Yahoo Finance (wymaga pakietu yfinance i internetu)."""
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"Brak danych dla {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # nowsze yfinance zwraca MultiIndex kolumn
        close = close.iloc[:, 0]
    return clean(close.rename(ticker))


def load_csv(path: str, date_col: str = "Date", price_col: str = "Close") -> pd.Series:
    """Wczytuje CSV z kolumną daty i ceny."""
    df = pd.read_csv(path, parse_dates=[date_col]).set_index(date_col).sort_index()
    return clean(df[price_col].rename(path))


def synthetic(n: int = 1500, seed: int = 0, drift: float = 0.0002, vol: float = 0.01,
              ar: float = 0.0, name: str = "SYNTH") -> pd.Series:
    """Syntetyczny szereg cen (log-zwroty AR(1) z szumem gaussowskim) — do testów i demo offline."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, vol, n)
    r = np.empty(n)
    r[0] = eps[0]
    for t in range(1, n):
        r[t] = drift + ar * r[t - 1] + eps[t]
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(100.0 * np.exp(np.cumsum(r)), index=idx, name=name)


def clean(s: pd.Series) -> pd.Series:
    """Usuwa braki i wartości niedodatnie (log-ceny muszą być skończone)."""
    s = s.astype(float).dropna()
    return s[s > 0]
