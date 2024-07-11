import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from app.learning.ab_test_model_factory import ABTestModelFactory, BinomialData

RANDOM_SEED = 4000
rng = np.random.default_rng(RANDOM_SEED)

# %config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")

plotting_defaults = dict(
    bins=50,
    kind="hist",
    textsize=10,
)


def plot_prior(model_factory: ABTestModelFactory, is_show: bool = False):
    with model_factory.create_model(data=[BinomialData(trials=1, successes=1)]):
        prior = pm.sample_prior_predictive(draws=100, return_inferencedata=False)

    fig, axs = plt.subplots(2, 1, figsize=(7, 7))
    az.plot_posterior(prior["reluplift_b"], ax=axs[0], **plotting_defaults)
    axs[0].set_title(
        f"B vs. A Rel Uplift Prior Predictive, {model_factory.priors}", fontsize=10
    )
    axs[0].axvline(x=0, color="red")
    if is_show:
        plt.show()
