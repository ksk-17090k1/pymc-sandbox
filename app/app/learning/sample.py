import pymc as pm
from pydantic import BaseModel

from app.learning.ab_test_model_factory import ABTestModelFactory, BinomialData


def sample_ab_test(
    model_factory: ABTestModelFactory, priors: BaseModel, data: list[BinomialData]
):
    with model_factory(priors=priors).create_model(data):
        pm.sample(draws=100)
