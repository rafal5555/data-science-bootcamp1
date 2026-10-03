"""Interfejs wiersza poleceń.

Przykłady:
    python -m finpredict backtest --preset kruszce --horizon 5
    python -m finpredict forecast --tickers EURPLN=X GC=F AAPL --horizon 5
    python -m finpredict backtest --csv ceny.csv --horizon 1
    python -m finpredict backtest --synthetic          # demo offline
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from . import data
from .evaluate import score, walk_forward
from .models import default_models


def _series(args) -> list[pd.Series]:
    if args.synthetic:
        return [data.synthetic(n=2000, seed=1, name="SYNTH_RW"),
                data.synthetic(n=2000, seed=2, ar=0.15, name="SYNTH_AR")]
    if args.csv:
        return [data.load_csv(p) for p in args.csv]
    tickers = list(args.tickers or [])
    for p in args.preset or []:
        tickers += data.PRESETS[p]
    if not tickers:
        sys.exit("Podaj --tickers, --preset, --csv lub --synthetic")
    out = []
    for t in tickers:
        try:
            out.append(data.load_yahoo(t, start=args.start))
        except Exception as exc:  # pojedynczy ticker nie powinien przerywać całego przebiegu
            print(f"[pomijam {t}] {exc}", file=sys.stderr)
    return out


def cmd_backtest(args) -> None:
    models = default_models(with_chronos=args.chronos)
    for s in _series(args):
        res = walk_forward(s, models, h=args.horizon, test_size=args.test_size, refit_every=args.refit_every)
        print(f"\n=== {s.name} | h={args.horizon} dni | {len(res)} prognoz poza próbą "
              f"({res.index[0].date()} – {res.index[-1].date()}) ===")
        print(score(res, args.horizon).round(4).to_string())
        if args.out:
            res.to_csv(f"{args.out}_{str(s.name).replace('=', '').replace('/', '_')}.csv")


def cmd_forecast(args) -> None:
    models = default_models(with_chronos=args.chronos)
    lo_q, hi_q = (1 - args.coverage) / 2, 1 - (1 - args.coverage) / 2
    for s in _series(args):
        # rezydua z walidacji kroczącej → empiryczny przedział (podejście w duchu split conformal)
        res = walk_forward(s, models, h=args.horizon, test_size=args.test_size, refit_every=args.refit_every)
        sc = score(res, args.horizon)
        logp = np.log(s.to_numpy(dtype=float))
        last_price = s.iloc[-1]
        print(f"\n=== {s.name} | ostatnia cena {last_price:.4f} ({s.index[-1].date()}) | horyzont {args.horizon} dni ===")
        rows = []
        for m in models:
            m.fit(logp, args.horizon)
            r = float(m.predict(logp, np.array([len(logp) - 1]))[0])
            resid = res["actual"] - res[m.name]
            lo, hi = r + resid.quantile(lo_q), r + resid.quantile(hi_q)
            rows.append({
                "model": m.name,
                "zwrot_%": 100 * (np.exp(r) - 1),
                "cena": last_price * np.exp(r),
                f"dolna_{int(args.coverage * 100)}%": last_price * np.exp(lo),
                f"gorna_{int(args.coverage * 100)}%": last_price * np.exp(hi),
                "R2oos_vs_RW": sc.loc[m.name, "R2oos_vs_RW"],
                "DM_p": sc.loc[m.name, "DM_p"],
            })
        print(pd.DataFrame(rows).set_index("model").round(4).to_string())
        best = sc.drop("RandomWalk")
        best = best[(best["R2oos_vs_RW"] > 0) & (best["DM_p"] < 0.05)]
        if best.empty:
            print("→ Żaden model nie pokonał istotnie błądzenia losowego w backteście; "
                  "traktuj prognozy punktowe jako niewiarygodne.")
        else:
            print(f"→ Istotnie lepsze od RW (p<0.05): {', '.join(best.index)}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="finpredict", description="Prognozy walut, kruszców i spółek z rzetelną walidacją.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("backtest", cmd_backtest), ("forecast", cmd_forecast)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--tickers", nargs="+", help="tickery Yahoo, np. EURPLN=X GC=F AAPL")
        p.add_argument("--preset", nargs="+", choices=list(data.PRESETS), help="gotowe zestawy")
        p.add_argument("--csv", nargs="+", help="pliki CSV z kolumnami Date, Close")
        p.add_argument("--synthetic", action="store_true", help="dane syntetyczne (offline)")
        p.add_argument("--start", default="2010-01-01")
        p.add_argument("--horizon", type=int, default=5, help="horyzont w dniach sesyjnych")
        p.add_argument("--test-size", type=int, default=750, help="liczba prognoz poza próbą")
        p.add_argument("--refit-every", type=int, default=21, help="co ile dni ponownie uczyć modele")
        p.add_argument("--chronos", action="store_true", help="dodaj Chronos-Bolt (wymaga chronos-forecasting)")
        if name == "backtest":
            p.add_argument("--out", help="prefiks plików CSV z prognozami")
        else:
            p.add_argument("--coverage", type=float, default=0.8, help="pokrycie przedziału")
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
