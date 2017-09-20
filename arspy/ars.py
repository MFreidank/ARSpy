from numpy import sign, log, unique, linspace, isinf
from numpy.random import rand
from arspy.hull import compute_hulls, evaluate_hulls, sample_upper_hull


def adaptive_rejection_sampling(logpdf,
                                a: float, b: float,
                                domain,
                                n_samples: int):

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
