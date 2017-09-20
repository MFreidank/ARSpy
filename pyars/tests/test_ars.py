from numpy import log, exp, allclose, asarray, sqrt
from numpy.random import seed
from subprocess import check_output

from os.path import dirname, realpath, join
from os import getenv

from pyars.ars import adaptive_rejection_sampling


def call_julia(name, a, b, n_samples, domain):
    a, b = str(a), str(b)
    n_samples = str(n_samples)
    domain = tuple(map(str, domain))

    reference_implementation_path = join(
        dirname(realpath(__file__)), "reference_implementation"
    )

    reference_script = join(reference_implementation_path, "reference.jl")

    julia_binary = getenv("PYARS_JULIA_BIN")

    cmd = [julia_binary, reference_script, name, a, b, n_samples, *domain, "nodebug"]
    print("CMD:", cmd)
    output = check_output(
        cmd
    ).decode()

    julia_result = asarray(list(map(float, output.strip("[]\n").split(","))))

    return julia_result


def gaussian(x, sigma=1):
    return log(exp(-x ** 2 / sigma))


def half_gaussian(x, sigma=3):
    return log(exp(-x ** 2 / sigma)) * (1 * (x <= 0) + 1e300 * (x > 0))


def relativistic_momentum_logpdf(p, m=1., c=1.):
    return -m * c ** 2 * sqrt(p ** 2 / (m ** 2 * c ** 2) + 1)


tests = {
    "1d-gaussian": {"name": "1d-gaussian",
                    "func": gaussian,
                    "a": -2, "b": 2,
                    "domain": (float("-inf"), float("inf")),
                    "n_samples": 20},
    "1d-half-gaussian": {"name": "1d-half-gaussian",
                         "func": half_gaussian,
                         "a": -2, "b": 0,
                         "domain": [float("-inf"), 0],
                         "n_samples": 20},
    # XXX: Add external function from arsDemo.m (third example)
    "relativistic_monte_carlo_logpdf": {
        "name": "relativistic_momentum_logpdf",
        "func": relativistic_momentum_logpdf,
        "a": -10.0, "b": 10.0,
        "domain": [float("-inf"), float("inf")],
        "n_samples": 20
    }

}


def _run(test_name):
    seed(1)
    input_dict = tests[test_name]

    name = input_dict["name"]
    a = input_dict["a"]
    b = input_dict["b"]
    domain = input_dict["domain"]
    n_samples = input_dict["n_samples"]

    julia_result = call_julia(
        name=name, a=a, b=b, domain=domain, n_samples=n_samples
    )

    logpdf = input_dict["func"]

    python_result = adaptive_rejection_sampling(
        logpdf=logpdf, a=a, b=b, domain=domain, n_samples=n_samples
    )

    assert(allclose(julia_result, python_result, atol=3e-01))


def test_gaussian():
    _run("1d-gaussian")


def test_half_gaussian():
    _run("1d-half-gaussian")

def test_relativistic_monte_carlo_logpdf():
    _run("relativistic_monte_carlo_logpdf")
