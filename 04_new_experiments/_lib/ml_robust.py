"""Mittag-Leffler E_alpha(-x) for 0<alpha<1, x>=0, via the spectral integral

    E_alpha(-x) = sin(a*pi)/(a*pi) * int_0^inf exp(-(x u)^{1/a}) / (u^2 + 2u cos(a*pi) + 1) du,

whose integrand is positive, so it avoids the cancellation that breaks the
truncated Taylor series at moderate x. Checked against E_{1/2}(-x) = exp(x^2) erfc(x).
"""
import math
import numpy as np
from scipy.integrate import quad


def _ml_scalar(x, alpha):
    if x < 1e-12:
        return 1.0
    sa = math.sin(alpha * math.pi)
    ca = math.cos(alpha * math.pi)
    ia = 1.0 / alpha

    def integrand(u):
        denom = u * u + 2.0 * u * ca + 1.0
        if denom < 1e-30:
            return 0.0
        xu = x * u
        if xu < 1e-300:
            return sa / denom
        try:
            xu_pow = xu ** ia
        except OverflowError:
            return 0.0
        if xu_pow > 700.0:  # exp(-xu_pow) underflows
            return 0.0
        return math.exp(-xu_pow) * sa / denom

    result, _ = quad(integrand, 0.0, np.inf, limit=200, epsabs=1e-12, epsrel=1e-10)
    return max(0.0, min(1.0, result / (alpha * math.pi)))


def mittag_leffler_neg(x, alpha):
    """E_alpha(-x), vectorised over x."""
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    return np.array([_ml_scalar(float(xi), float(alpha)) for xi in x])


ml_neg = mittag_leffler_neg
