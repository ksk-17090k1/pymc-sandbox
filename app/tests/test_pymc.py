from typing import Dict, List, Union

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from pydantic import BaseModel
from scipy.stats import bernoulli, expon

from app.learning.ab_test_model_factory import (
    ABTestModelFactory,
    BetaPrior,
    BinomialData,
)
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


def test_sample_ab_test():
    sample_ab_test(
        ABTestModelFactory,
        BetaPrior(alpha=100, beta=100),
        [BinomialData(trials=1, successes=1)],
    )
