# finpredict — prognozy walut, kruszców i spółek

Aplikacja CLI prognozująca h-dniowy zwrot (i wynikającą z niego cenę) dla kursów walut,
metali szlachetnych/przemysłowych i akcji. Najważniejsza cecha: **każda prognoza jest
oceniana uczciwie poza próbą i porównywana z błądzeniem losowym**, bo w finansach to jest
benchmark, który najczęściej wygrywa.

## Szybki start

```bash
pip install -r requirements.txt
python -m finpredict backtest --preset waluty kruszce spolki --horizon 5
python -m finpredict forecast --tickers EURPLN=X GC=F PKO.WA --horizon 5
python -m finpredict backtest --csv moje_ceny.csv --horizon 1   # CSV: kolumny Date, Close
python -m finpredict backtest --synthetic                         # demo bez internetu
python -m pytest -q tests
```

Zestawy (`--preset`): `waluty` (EURUSD, EURPLN, USDPLN, GBPUSD, USDJPY),
`kruszce` (złoto, srebro, platyna, pallad, miedź — kontrakty futures), `spolki` (AAPL, MSFT, NVDA, PKO, PKN).

## Co robi

| Element | Opis | Źródło |
|---|---|---|
| `RandomWalk` | prognoza zwrotu = 0; benchmark | Meese & Rogoff (1983), *J. Int. Econ.* 14 |
| `HistMean` | średni historyczny dryf | Campbell & Thompson (2008), *RFS* 21(4) |
| `DLinear` | rozkład okna na trend + resztę, dwie warstwy liniowe, prognoza bezpośrednia | Zeng i in., *AAAI 2023*, [arXiv:2205.13504](https://arxiv.org/abs/2205.13504) |
| `GBoost` | drzewa wzmacniane na cechach momentum/zmienności | Gu, Kelly & Xiu (2020), *RFS* 33(5) |
| `Chronos` (opcja `--chronos`) | prekonfigurowany model bazowy, zero-shot | Ansari i in., *TMLR 2024*, [arXiv:2403.07815](https://arxiv.org/abs/2403.07815) |
| walidacja krocząca | ponowne uczenie co `--refit-every` dni, tylko na danych z przeszłości | — |
| test Diebolda–Mariano | czy przewaga nad RW jest istotna statystycznie (HAC, poprawka HLN) | Diebold & Mariano (1995), *JBES* 13(3); Harvey, Leybourne & Newbold (1997), *IJF* 13(2) |
| R²oos | 1 − SSE(model)/SSE(RW); > 0 oznacza poprawę | Campbell & Thompson (2008) |

`forecast` dodatkowo podaje przedział (domyślnie 80%) z empirycznych kwantyli błędów z backtestu
i informuje, czy którykolwiek model istotnie (p<0,05) pokonał błądzenie losowe.

## Ważne zastrzeżenia (proszę przeczytać)

1. **arXiv nie jest recenzowany.** To serwer preprintów. Dlatego do implementacji wybrano wyłącznie
   prace z arXiv, które **przeszły recenzję i zostały opublikowane** (AAAI 2023, TMLR 2024).
   PatchTST (Nie i in., *ICLR 2023*, [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)) nie został
   zaimplementowany — wymaga PyTorcha, a DLinear w tej samej linii badań jest prostszy i konkurencyjny.
   Najnowszy duży benchmark dla futures (Saly-Kaufmann i in., 2026, [arXiv:2603.01820](https://arxiv.org/abs/2603.01820))
   to — według stanu sprawdzonego przy tworzeniu — preprint, więc służy tylko jako kontekst: wskazuje, że
   hybrydy typu VSN+LSTM osiągają wyższe Sharpe niż modele liniowe w zadaniu alokacji pozycji.
2. **„Najnowsze” ≠ „najlepsze dla finansów”.** Modele z benchmarków ogólnych szeregów czasowych
   (pogoda, energia, ruch) nie przenoszą automatycznie przewagi na zwroty finansowe, gdzie stosunek
   sygnału do szumu jest bardzo niski.
3. **Brak sygnału jest typowym wynikiem.** Jeśli `DM_p` > 0,05, model nie jest lepszy od „jutro tak jak dziś”.
   Trafność kierunku ~50–55% bez istotności nie jest przewagą.
4. **Testowanie wielu tickerów/horyzontów/modeli** zwiększa ryzyko fałszywych odkryć — przy wielu porównaniach
   pojedyncze p<0,05 jest spodziewane przez przypadek.
5. **Przedziały** nie mają gwarancji pokrycia — rynki są niestacjonarne (zmiany reżimu zmienności).
6. Brak kosztów transakcyjnych, poślizgu i podatków. **To narzędzie badawcze, nie rekomendacja inwestycyjna.**

## Struktura

```
finpredict/data.py      – Yahoo Finance / CSV / dane syntetyczne, zestawy tickerów
finpredict/models.py    – RandomWalk, HistMean, DLinear, GBoost, Chronos
finpredict/evaluate.py  – walk-forward, metryki, test Diebolda–Mariano
finpredict/cli.py       – komendy backtest / forecast
tests/                  – m.in. test braku podglądania przyszłości
```
