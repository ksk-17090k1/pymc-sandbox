# %%
from typing import Dict, List, Union

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from pydantic import BaseModel
from scipy.stats import bernoulli, expon

# %%
RANDOM_SEED = 4000
rng = np.random.default_rng(RANDOM_SEED)

# %config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")

plotting_defaults = dict(
    bins=50,
    kind="hist",
    textsize=10,
)
print("start!")

# %%


class BetaPrior(BaseModel):
    alpha: float
    beta: float


class BinomialData(BaseModel):
    trials: int
    successes: int


class ConversionModelTwoVariant(BaseModel):
    priors: BetaPrior

    def create_model(self, data: List[BinomialData]) -> pm.Model:
        trials = [d.trials for d in data]
        successes = [d.successes for d in data]
        with pm.Model() as model:
            p = pm.Beta("p", alpha=self.priors.alpha, beta=self.priors.beta, shape=2)
            obs = pm.Binomial("y", n=trials, p=p, shape=2, observed=successes)
            reluplift = pm.Deterministic("reluplift_b", p[1] / p[0] - 1)
        return model


weak_prior = ConversionModelTwoVariant(priors=BetaPrior(alpha=100, beta=100))

strong_prior = ConversionModelTwoVariant(priors=BetaPrior(alpha=10000, beta=10000))

# %%
with weak_prior.create_model(
    data=[BinomialData(trials=1, successes=1), BinomialData(trials=1, successes=1)]
):
    weak_prior_predictive = pm.sample_prior_predictive(
        samples=10000, return_inferencedata=False
    )

with strong_prior.create_model(
    data=[BinomialData(trials=1, successes=1), BinomialData(trials=1, successes=1)]
):
    strong_prior_predictive = pm.sample_prior_predictive(
        samples=10000, return_inferencedata=False
    )

fig, axs = plt.subplots(2, 1, figsize=(7, 7))
az.plot_posterior(weak_prior_predictive["reluplift_b"], ax=axs[0], **plotting_defaults)
axs[0].set_title(
    f"B vs. A Rel Uplift Prior Predictive, {weak_prior.priors}", fontsize=10
)
axs[0].axvline(x=0, color="red")
az.plot_posterior(
    strong_prior_predictive["reluplift_b"], ax=axs[1], **plotting_defaults
)
axs[1].set_title(
    f"B vs. A Rel Uplift Prior Predictive, {strong_prior.priors}", fontsize=10
)
axs[1].axvline(x=0, color="red")

# %%


def generate_binomial_data(
    variants: List[str], true_rates: List[str], samples_per_variant: int = 100000
) -> pd.DataFrame:
    data = {}
    for variant, p in zip(variants, true_rates):
        data[variant] = bernoulli.rvs(p, size=samples_per_variant)
    agg = (
        pd.DataFrame(data)
        .aggregate(["count", "sum"])
        .rename(index={"count": "trials", "sum": "successes"})
    )
    return agg


# Example generated data
generate_binomial_data(["A", "B"], [0.23, 0.23])


# %%


def run_scenario_twovariant(
    variants: List[str],
    true_rates: List[float],
    samples_per_variant: int,
    weak_prior: BetaPrior,
    strong_prior: BetaPrior,
) -> None:
    generated = generate_binomial_data(variants, true_rates, samples_per_variant)
    data = [BinomialData(**generated[v].to_dict()) for v in variants]
    with ConversionModelTwoVariant(priors=weak_prior).create_model(data):
        trace_weak = pm.sample(draws=5000)
    with ConversionModelTwoVariant(priors=strong_prior).create_model(data):
        trace_strong = pm.sample(draws=5000)

    true_rel_uplift = true_rates[1] / true_rates[0] - 1

    fig, axs = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    az.plot_posterior(
        trace_weak.posterior["reluplift_b"], ax=axs[0], **plotting_defaults
    )
    axs[0].set_title(
        f"True Rel Uplift = {true_rel_uplift:.1%}, {weak_prior}", fontsize=10
    )
    axs[0].axvline(x=0, color="red")
    az.plot_posterior(
        trace_strong.posterior["reluplift_b"], ax=axs[1], **plotting_defaults
    )
    axs[1].set_title(
        f"True Rel Uplift = {true_rel_uplift:.1%}, {strong_prior}", fontsize=10
    )
    axs[1].axvline(x=0, color="red")
    fig.suptitle("B vs. A Rel Uplift")
    return trace_weak, trace_strong


trace_weak, trace_strong = run_scenario_twovariant(
    variants=["A", "B"],
    true_rates=[0.23, 0.23],
    samples_per_variant=100000,
    weak_prior=BetaPrior(alpha=100, beta=100),
    strong_prior=BetaPrior(alpha=10000, beta=10000),
)
# %%

run_scenario_twovariant(
    variants=["A", "B"],
    true_rates=[0.21, 0.23],
    samples_per_variant=100000,
    weak_prior=BetaPrior(alpha=100, beta=100),
    strong_prior=BetaPrior(alpha=10000, beta=10000),
)
# %%
