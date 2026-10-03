import numpy as np
import pytest

from finpredict import data
from finpredict.cli import main
from finpredict.evaluate import diebold_mariano, score, walk_forward
from finpredict.models import DLinear, GradientBoosting, HistoricalMean, RandomWalk


@pytest.mark.parametrize("model", [DLinear(), GradientBoosting(max_iter=20), HistoricalMean()])
def test_no_lookahead(model):
    """Zmiana przyszłych cen nie może zmienić prognozy z punktu o."""
    logp = np.log(data.synthetic(n=800, seed=5).to_numpy())
    o = np.array([600, 650])
    model.fit(logp[:601], h=5)
    before = model.predict(logp, o[:1])
    future = logp.copy()
    future[601:] += np.random.default_rng(0).normal(0, 0.5, len(future) - 601)
    assert np.allclose(before, model.predict(future, o[:1]))


def test_dlinear_finds_real_signal():
    s = data.synthetic(n=2000, seed=3, ar=0.3)
    res = walk_forward(s, [RandomWalk(), DLinear()], h=1, test_size=500, refit_every=50)
    sc = score(res, h=1)
    assert sc.loc["DLinear", "R2oos_vs_RW"] > 0.03
    assert sc.loc["DLinear", "DM_p"] < 0.05


def test_no_false_signal_on_random_walk():
    s = data.synthetic(n=1500, seed=11)
    res = walk_forward(s, [RandomWalk(), DLinear()], h=1, test_size=500, refit_every=50)
    assert score(res, h=1).loc["DLinear", "DM_p"] > 0.01


def test_diebold_mariano_direction():
    rng = np.random.default_rng(0)
    e_good, e_bad = rng.normal(0, 1, 500), rng.normal(0, 1.5, 500)
    stat, p = diebold_mariano(e_good, e_bad, h=1)
    assert stat < 0 and p < 0.01


def test_walk_forward_targets_are_realised_returns():
    s = data.synthetic(n=1200, seed=2)
    res = walk_forward(s, [RandomWalk()], h=3, test_size=100, refit_every=25)
    logp = np.log(s)
    i = s.index.get_loc(res.index[0])
    assert np.isclose(res["actual"].iloc[0], logp.iloc[i + 3] - logp.iloc[i])
    assert s.index.get_loc(res.index[-1]) == len(s) - 1 - 3


def test_cli_smoke(capsys):
    main(["forecast", "--synthetic", "--horizon", "2", "--test-size", "100", "--refit-every", "50"])
    out = capsys.readouterr().out
    assert "SYNTH_RW" in out and "DLinear" in out
