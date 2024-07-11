# %%

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from pydantic import BaseModel

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


# %%

# モデルの定義


class BetaPrior(BaseModel):
    alpha: float
    beta: float


class BinomialData(BaseModel):
    trials: int
    successes: int


class ScoutModelFactory(BaseModel):
    priors: BetaPrior

    def create_model(self, data: list[BinomialData]) -> pm.Model:
        trials = [d.trials for d in data]
        successes = [d.successes for d in data]
        with pm.Model() as model:
            p = pm.Beta("p", alpha=self.priors.alpha, beta=self.priors.beta, shape=2)
            pm.Binomial("y", n=trials, p=p, shape=2, observed=successes)
            pm.Deterministic("delta", p[0] - p[1])
        return model


prior = ScoutModelFactory(priors=BetaPrior(alpha=3, beta=60))

# %%

# 事前分布の確認
# だいたい中央値が5%くらいの分布に調整する
with prior.create_model(data=[BinomialData(trials=1, successes=1)]):
    prior_predictive = pm.sample_prior_predictive(
        samples=10000, return_inferencedata=False
    )

fig, ax = plt.subplots(figsize=(7, 7))

az.plot_posterior(prior_predictive["p"], ax=ax, **plotting_defaults)
ax.set_title(f"Prior Predictive, {prior.priors}", fontsize=10)

# %%

scout_data = pd.DataFrame(
    index=["trials", "successes"],
    data={
        "AI": [10_460, 380],
        "Human": [20_474, 590],
    },
)
scout_data
# %%

# サンプリング実行


def run_abtest(
    prior: BetaPrior,
    data: list[BinomialData],
) -> None:
    with ScoutModelFactory(priors=prior).create_model(data):
        trace = pm.sample(draws=5000)

    fig, ax = plt.subplots(figsize=(7, 7))
    az.plot_posterior(trace.posterior["delta"], ax=ax, **plotting_defaults)
    ax.set_title(f"prior: {prior}", fontsize=10)
    ax.axvline(x=0, color="red")

    return trace


trace = run_abtest(
    prior=BetaPrior(alpha=10, beta=10),
    data=[BinomialData(**scout_data[v].to_dict()) for v in ["AI", "Human"]],
)

# %%

az.plot_posterior(
    trace,
    hdi_prob=0.80,
    #   ref_val=0,
    # var_names=var_names,
    figsize=(10, 5),
)


# %%
