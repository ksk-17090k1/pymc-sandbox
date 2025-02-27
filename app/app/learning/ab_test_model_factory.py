import arviz as az
import numpy as np
import pymc as pm
from pydantic import BaseModel

RANDOM_SEED = 4000
rng = np.random.default_rng(RANDOM_SEED)

# %config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")


class BetaPrior(BaseModel):
    alpha: float
    beta: float


class BinomialData(BaseModel):
    trials: int
    successes: int


class ABTestModelFactory(BaseModel):
    priors: BetaPrior

    def create_model(self, data: list[BinomialData]) -> pm.Model:
        trials = [d.trials for d in data]
        successes = [d.successes for d in data]
        with pm.Model() as model:
            p = pm.Beta("p", alpha=self.priors.alpha, beta=self.priors.beta, shape=2)
            pm.Binomial("y", n=trials, p=p, shape=2, observed=successes)
            pm.Deterministic("reluplift_b", p[1] / p[0] - 1)
        return model
