from numpy import log, exp, allclose, sqrt
from numpy.random import seed
import numpy as np

from os.path import dirname, realpath, join

from arspy.ars import adaptive_rejection_sampling


data_file = "{}/ars_{{}}.npy".format(
    join(dirname(realpath(__file__)), "reference_data", "ars_data")
).format


def gaussian(x, sigma=1):
    return log(exp(-x ** 2 / sigma))


def half_gaussian(x, sigma=3):
    return log(exp(-x ** 2 / sigma)) * (1 * (x <= 0) + 1e300 * (x > 0))


def relativistic_momentum_logpdf(p, m=1., c=1.):
    return -m * c ** 2 * sqrt(p ** 2 / (m ** 2 * c ** 2) + 1)


tests = {
    "1d-gaussian": {"name": "1d-gaussian",
                    "data": data_file("gaussian"),
                    "func": gaussian,
                    "a": -2, "b": 2,
                    "domain": (float("-inf"), float("inf")),
                    "n_samples": 20},
    "1d-half-gaussian": {"name": "1d-half-gaussian",
                         "data": data_file("half_gaussian"),
                         "func": half_gaussian,
                         "a": -2, "b": 0,
                         "domain": [float("-inf"), 0],
                         "n_samples": 20},
    "relativistic_monte_carlo_logpdf": {
        "name": "relativistic_momentum_logpdf",
        "data": data_file("relativistic_logpdf"),
        "func": relativistic_momentum_logpdf,
        "a": -10.0, "b": 10.0,
        "domain": [float("-inf"), float("inf")],
        "n_samples": 20
    }

}


def _run(test_name):
    seed(1)
    input_dict = tests[test_name]

    # name = input_dict["name"]
    a = input_dict["a"]
    b = input_dict["b"]
    domain = input_dict["domain"]
    n_samples = input_dict["n_samples"]

    logpdf = input_dict["func"]

    python_result = adaptive_rejection_sampling(
        logpdf=logpdf, a=a, b=b, domain=domain, n_samples=n_samples
    )

    # load old result computed by other implementation (julia)
    julia_result = np.load(input_dict["data"])

    assert(allclose(julia_result, python_result, atol=3e-01))


def test_gaussian():
    _run("1d-gaussian")


def test_half_gaussian():
    _run("1d-half-gaussian")


def test_relativistic_monte_carlo_logpdf():
    _run("relativistic_monte_carlo_logpdf")
