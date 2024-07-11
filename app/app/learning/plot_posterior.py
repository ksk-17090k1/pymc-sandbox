import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from arviz import InferenceData

RANDOM_SEED = 4000
rng = np.random.default_rng(RANDOM_SEED)

# %config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")

plotting_defaults = dict(
    bins=50,
    kind="hist",
    textsize=10,
)


def plot_posterior(trace: InferenceData, is_show: bool = False):
    fig, axs = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    az.plot_posterior(trace.posterior["reluplift_b"], ax=axs[0], **plotting_defaults)
    axs[0].set_title("sub title", fontsize=10)
    axs[0].axvline(x=0, color="red")
    if is_show:
        plt.show()
