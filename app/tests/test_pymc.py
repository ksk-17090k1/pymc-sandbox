import sys

import arviz as az
import numpy as np
import pymc as pm
import pytest

sys.path.append(".")

from app.learning.ab_test_model_factory import (
    ABTestModelFactory,
    BetaPrior,
    BinomialData,
)
from app.learning.plot_posterior import plot_posterior
from app.learning.plot_prior import plot_prior
from app.learning.sample import sample_ab_test

RANDOM_SEED = 4000
rng = np.random.default_rng(RANDOM_SEED)

# %config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")


def test_ab_test_model_factory():
    ABTestModelFactory(priors=BetaPrior(alpha=100, beta=100))
    ABTestModelFactory(priors=BetaPrior(alpha=10000, beta=10000))


def test_plot_prior():
    model_factory = ABTestModelFactory(priors=BetaPrior(alpha=100, beta=100))
    plot_prior(model_factory)


@pytest.mark.skip
def test_sample_ab_test():
    sample_ab_test(
        ABTestModelFactory,
        BetaPrior(alpha=100, beta=100),
        [BinomialData(trials=1, successes=1)],
    )


def test_plot_posterior(is_show: bool = False):
    model_factory = ABTestModelFactory(priors=BetaPrior(alpha=100, beta=100))
    with model_factory.create_model(data=[BinomialData(trials=1, successes=1)]):
        trace = pm.sample()

    plot_posterior(trace, is_show)


if __name__ == "__main__":
    # test_ab_test_model_factory()
    # test_plot_prior()
    # test_sample_ab_test()
    test_plot_posterior(is_show=True)
