import great_expectations as ge
import pandas as pd
import pytest


def pytest_addoption(parser):
    """Add option to specify dataset location when executing tests from CLI.
    Ex: pytest --dataset-loc=checkpoints/data.csv tests/data --verbose --disable-warnings
    """
    parser.addoption(
        "--dataset-loc", action="store", default=None, help="Dataset location."
    )


@pytest.fixture()
def df(request):
    dataset_loc = request.config.getoption("--dataset-loc")
    df = dat
    return df
