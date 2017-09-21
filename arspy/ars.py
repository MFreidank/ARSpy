"""
This module contains a python/numpy implementation
of Adaptive Rejection Sampling (ARS) introduced
by Gilks and Wild in 1992.

Our implementation considers both lower and upper hull
as introduced in the original paper. This allows us
to extract larger amounts of samples at a time
in a stable way. Furthermore, we do *not* require the *derivative*
of the logpdf as input to our sampler, only the logpdf itself.

Our code is a port of an original matlab code in pmtk3 by Daniel Eaton (danieljameseaton@gmail.com) and compared to an open-source julia port (by Levi Boyles) of the same matlab function for testing purposes.
"""
from numpy import sign, log, unique, linspace, isinf
from numpy.random import rand
from arspy.hull import compute_hulls, evaluate_hulls, sample_upper_hull
from typing import Tuple

__all__ = (
    "adaptive_rejection_sampling",
)


def adaptive_rejection_sampling(logpdf: callable,
                                a: float, b: float,
                                domain: Tuple[float, float],
                                n_samples: int):
    """
    Adaptive rejection sampling samples exactly (all samples are i.i.d) and efficiently from any univariate log-concave distribution. The basic idea is to successively determine an envelope of straight-line segments to construct an increasingly accurate approximation of the logarithm.
    It does not require any normalization of the target distribution.

    Parameters
    ----------
    logpdf: callable
        Univariate function that computes :math:`log(f(u))`
        for a given :math:`u`, where :math:`f(u)` is proportional
        to the target density to sample from.

    a: float
        Lower starting point used to initialize the hulls.
        Must lie in the domain of the logpdf and it
        must hold: :math:`a < b`.

    b: float
        Upper starting point used to initialize the hulls.
        Must lie in the domain of the logpdf and it
        must hold: :math:`a < b`.

    domain : Tuple[float, float]
        Domain of `logpdf`.
        May be unbounded on either or both sides,
        in which case `(float("-inf"), float("inf"))`
        would be passed.
        If this domain is unbounded to the left,
        the derivative of the logpdf
        for x<= a must be positive.
        If this domain is unbounded to the right                  the derivative of the logpdf for x>=b
        must be negative.

    n_samples: int
        Number of samples to draw.

    Returns
    ----------
    samples : list
        A list of samples drawn from the
        target distribution :math:`f`
        with the given `logpdf`.

    Examples
    ----------
    Sampling from a simple gaussian, adaptive rejection sampling style.
    We use the logpdf of a standard gaussian and this small code snippet
    demonstrates that our sample approximation accurately approximates the mean:

    >>> from math import isclose
    >>> from numpy import log, exp, mean
    >>> gaussian_logpdf = lambda x, sigma=1: log(exp(-x ** 2 / sigma))
    >>> a, b = -2, 2  # a < b must hold
    >>> domain = (float("-inf"), float("inf"))
    >>> n_samples = 10000
    >>> samples = adaptive_rejection_sampling(logpdf=gaussian_logpdf, a=a, b=b, domain=domain, n_samples=n_samples)
    >>> isclose(mean(samples), 0.0, abs_tol=1e-02)
    True

    """
    assert(hasattr(logpdf, "__call__"))
    assert(len(domain) == 2), "Domain must be two-element iterable."
    assert(domain[1] >= domain[0]), "Invalid domain, it must hold: domain[1] >= domain[0]."
    assert(n_samples >= 0), "Number of samples must be >= 0."

    if a >= b or isinf(a) or isinf(b) or a < domain[0] or b > domain[1]:
        raise ValueError("invalid a and b")

    n_derivative_steps = 1e-3 * (b - a)

    S = (a, a + n_derivative_steps, b - n_derivative_steps, b)

    if domain[0] == float("-inf"):
        # ensure positive derivative at 'a'
        derivative_sign = sign(logpdf(a + n_derivative_steps) - logpdf(a))
        positive_derivative = derivative_sign > 0

        assert(positive_derivative), "derivative at 'a' must be positive, since the domain is unbounded to the left"

    if domain[1] == float("inf"):
        # ensure negative derivative at 'b'
        derivative_sign = sign(logpdf(b) - logpdf(b - n_derivative_steps))
        negative_derivative = derivative_sign < 0

        assert(negative_derivative), "derivative at 'b' must be negative, since the domain is unbounded to the right"

    # initialize a mesh on which to create upper & lower hulls
    n_initial_mesh_points = 3

    S = unique(
        (S[0], *(linspace(S[1], S[2], num=n_initial_mesh_points + 2)), S[3])
    )

    fS = tuple(logpdf(s) for s in S)

    lower_hull, upper_hull = compute_hulls(S=S, fS=fS, domain=domain)

    samples = []

    while len(samples) < n_samples:

        mesh_changed = False

        x = sample_upper_hull(upper_hull)

        lh_val, uh_val = evaluate_hulls(x, lower_hull, upper_hull)

        U = rand()

        if log(U) <= lh_val - uh_val:
            # accept u is below lower bound
            samples.append(x)

        elif log(U) <= logpdf(x) - uh_val:
            # accept, u is between lower bound and f
            samples.append(x)

            mesh_changed = True
        else:
            # reject, u is between f and upper_bound
            mesh_changed = True

        if mesh_changed:
            S = sorted([*S, x])
            fS = tuple(logpdf(s) for s in S)
            lower_hull, upper_hull = compute_hulls(S=S, fS=fS, domain=domain)
    return samples
