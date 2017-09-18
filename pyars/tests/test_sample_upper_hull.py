# XXX: Make seed variable
# This seems to work
"""
from numpy import allclose, asarray
from numpy.random import seed
from subprocess import check_output

from os.path import dirname, realpath, join
from os import getenv

from pyars.hull import HullNode, sample_upper_hull

from hypothesis import given
from hypothesis.strategies import integers


def call_julia(n_samples=1):
    reference_implementation_path = join(
        dirname(realpath(__file__)), "reference_implementation"
    )

    reference_script = join(
        reference_implementation_path, "reference_sample_upper_hull.jl"
    )

    julia_binary = getenv("PYARS_JULIA_BIN")

    output = check_output(
        [julia_binary, reference_script, str(n_samples)]
    ).decode()

    julia_result = asarray(list(map(float, output.strip("[]\n").split(","))))

    return julia_result


@given(integers(min_value=10, max_value=10))
def test_valid_inputs(n_samples):
    julia_result = call_julia(
        n_samples=n_samples,
    )

    def hn(m, b, left, right, pr):
        return HullNode(m=m, b=b, left=left, right=right, pr=pr)

    upper_hull = [
        hn(3.995999999999996, 3.991999999999992, float("-inf"), -2.0, 0.0018422607166365945),
        hn(2.9939999999999998, 1.9920079999999998, -2.0, -1.996, 2.9742698274522277e-5),
        hn(3.995999999999996, 3.9919999999999924, -1.996, -1.331554369579719, 0.02475975522918407),
        hn(0.998, 0.0, -1.331554369579719, -0.998, 0.04211918296053056),
        hn(2.9939999999999998, 1.9920079999999998, -0.998, -0.499, 0.17130889715661873),
        hn(-0.998, 0.0, -0.499, 0.0, 0.2599401612387555),
        hn(0.998, 0.0, 0.0, 0.499, 0.2599401612387555),
        hn(-2.9939999999999998, 1.9920079999999998, 0.499, 0.998, 0.17130889715661873),
        hn(-0.998, 0.0, 0.998, 1.331554369579719, 0.04211918296053056),
        hn(-3.995999999999996, 3.9919999999999924, 1.331554369579719, 1.996, 0.02475975522918407),
        hn(-2.9939999999999998, 1.9920079999999998, 1.996, 2.0, 2.9742698274522277e-5),
        hn(-3.995999999999996, 3.991999999999992, 2.0, float("inf"), 0.0018422607166365945)
    ]
    seed(1)

    python_result = asarray([sample_upper_hull(upper_hull) for _ in range(n_samples)])

    assert(allclose(julia_result, python_result, atol=1e-02))
"""
