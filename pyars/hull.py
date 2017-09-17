from numpy import asarray, isinf, isnan, spacing as eps, log, exp, cumsum
from numpy.random import rand
from pyars.probability_utils import exp_normalize

# XXX: Globally check "isinf" usage: it also returns true for -inf, so
# make sure this is desired wherever we use it


class HullNode(object):
    def __init__(self, m, b, left, right, pr=None):
        self.m, self.b = m, b
        self.left, self.right = left, right
        self.pr = pr


def compute_hulls(S, fS, domain):
    assert(len(S) == len(fS))

    lower_hull = []

    for li in range(len(S) - 1):
        m = (fS[li + 1] - fS[li]) / (S[li + 1] - S[li])
        b = fS[li] - m * S[li]
        left = S[li]
        right = S[li + 1]
        lower_hull.append(HullNode(m=m, b=b, left=left, right=right))

    # compute upper piecewise-linear hull

    # NOTE: Use this in final assertion about length (check usage in julia code)
    n_upper_segments = 2 * (len(S) - 2) + isinf(domain[0]) + isinf(domain[1])

    i = 0

    upper_hull = []

    if isinf(domain[0]):
        # first line (from -infinity)
        m = (fS[1] - fS[0]) / (S[1] - S[0])
        b = fS[0] - m * S[0]
        pr = compute_segment_log_prob(float("-inf"), S[0], m, b)

        i += 1
        upper_hull.append(
            HullNode(m=m, b=b, pr=pr, left=float("-inf"), right=S[0])
        )

    # second line
    m = (fS[2] - fS[1]) / S[2] - S[1]
    b = fS[1] - m * S[1]
    pr = compute_segment_log_prob(S[0], S[1], m, b)

    i += 1
    upper_hull.append(HullNode(m=m, b=b, pr=pr, left=S[0], right=S[1]))

    # interior lines
    for li in range(1, len(S) - 2):
        m1 = (fS[li] - fS[li - 1]) / S[li] - S[li - 1]
        b1 = fS[li] - m1 * S[li]

        m2 = (fS[li + 2] - fS[li + 1]) / (S[li + 2] - S[li + 1])
        b2 = fS[li + 1] - m2 * S[li + 1]

        if isinf(m1) and isinf(m2):
            raise ValueError("both hull slopes are infinite!")

        dx1 = S[li] - S[li - 1]
        df1 = fS[li] - fS[li - 1]
        dx2 = S[li + 2] - S[li + 1]
        df2 = fS[li + 2] - fS[li + 1]

        f1 = fS[li]
        f2 = fS[li + 1]
        x1 = S[li]
        x2 = S[li + 1]

        # more numerically stable than above
        ix = ((f1 * dx1 - df1 * x1) * dx2 - (f2 * dx2 - df2 * x2) * dx1) / (df2 * dx1 - df1 * dx2)

        if isinf(m1) or abs(m1 - m2) < 10.0 ** 8 * eps(m1):
            ix = S[li]
            pr1 = float("-inf")

            pr2 = compute_segment_log_prob(ix, S[li + 1], m2, b2)
        elif isinf(m2):
            ix = S[li + 1]
            pr1 = compute_segment_log_prob(S[li], ix, m1, b2)
            pr2 = float("-inf")
        else:
            if isinf(ix):
                raise ValueError("Non finite intersection")

            if abs(ix - S[li]) < 10.0 ** 12 * eps(S[li]):
                ix = S[li]
            elif abs(ix - S[li + 1]) < 10.0 ** 12 * eps(S[li + 1]):
                ix = S[li + 1]

            if ix < S[li] or ix > S[li + 1]:
                raise ValueError("Intersection out of bounds -- logpdf is not concave")

            pr1 = compute_segment_log_prob(S[li], ix, m1, b1)
            pr2 = compute_segment_log_prob(ix, S[li + 1], m2, b2)

            i += 1
            upper_hull.append(HullNode(m=m1, b=b1, pr=pr1, left=S[li], right=ix))

            i += 1
            upper_hull.append(HullNode(m=m2, b=b2, pr=pr2, left=ix, right=S[li + 1]))

    # second last line
    m = (fS[-2] - fS[-3]) / (S[-2] - S[-3])
    b = fS[-2] - m * S[-2]
    pr = compute_segment_log_prob(S[-2], S[-1], m, b)
    upper_hull.append(HullNode(m=m, b=b, pr=pr, left=S[-2], right=S[-1]))

    i += 1

    if isinf(domain[1]):
        # last line (to infinity)
        m = (fS[-1] - fS[-1 - 1]) / (S[-1] - S[-1 - 1])
        b = fS[-1] - m * S[-1]
        pr = compute_segment_log_prob(S[-1], float("inf"), m, b)

        i += 1
        upper_hull.append(HullNode(m=m, b=b, pr=pr, left=S[-1], right=float("inf")))

    Z = sum(node.pr for node in upper_hull)

    for node in upper_hull:
        node.pr /= Z

    probs = exp_normalize(asarray([node.pr for node in upper_hull]))

    for node, prob in zip(upper_hull, probs):
        node.pr = prob

    return lower_hull, upper_hull


def compute_segment_log_prob(l, r, m, b):
    if l == float("-inf"):
        return -log(m) + m * r + b

    elif r == float("inf"):
        return -log(m) + m * l + b
    M = max(m * r + b, m * l + b)

    return -log(abs(m)) + log(abs(exp(m * r + b - M) - exp(m * l + b - M))) + M


def sample_upper_hull(upper_hull):
    cdf = cumsum([node.pr for node in upper_hull])

    # randomly choose a line segment
    U = rand()

    node = next(node for node, cdf_value in zip(upper_hull, cdf) if U < cdf_value)

    # sample along that line segment
    U = rand()

    m, left, right = node.m, node.left, node.right

    M = max(m * right, m * left)
    x = (log(U * (exp(m * right - M) - exp(m * left - M)) + exp(m * left - M)) + M) / m

    assert(x >= left and x <= right)

    if isinf(x) or isnan(x):
        raise ValueError("sampled an infinite or 'nan' x")

    return x


def evaluate_hulls(x, lower_hull, upper_hull):
    lh_val = 0.0

    # lower bound
    if x < min(node.left for node in lower_hull):
        lh_val = float("-inf")
    elif x > max(node.right for node in lower_hull):
        lh_val = float("-inf")
    else:
        node = next(node for node in lower_hull if node.left <= x <= node.right)
        lh_val = node.m * x + node.b

    uh_val = 0.0
    # upper bound
    node = next(node for node in upper_hull if node.left <= x <= node.right)
    uh_val = node.m * x + node.b

    return lh_val, uh_val
