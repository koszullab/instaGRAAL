#!/usr/bin/env python3

"""
Constrained multivariate Levenberg-Marquardt optimization

An updated version of this file can be found at
https://github.com/jjhelmus/leastsqbound-scipy

The version here has known bugs which have been
fixed above, proceed at your own risk.

- Jonathan J. Helmus (jjhelmus@gmail.com)
"""

from scipy.optimize import leastsq
import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import cholesky as _cholesky


def internal2external_grad(xi, bounds):
    """
    Calculate the internal to external gradiant

    Calculates the partial of external over internal

    """

    ge = np.empty_like(xi)

    for i, (v, bound) in enumerate(zip(xi, bounds, strict=False)):
        a = bound[0]  # minimum
        b = bound[1]  # maximum

        if a is None and b is None:  # No constraints
            ge[i] = 1.0

        elif b is None:  # only min
            ge[i] = v / np.sqrt(v**2 + 1)

        elif a is None:  # only max
            ge[i] = -v / np.sqrt(v**2 + 1)

        else:  # both min and max
            ge[i] = (b - a) * np.cos(v) / 2.0

    return ge


def i2e_cov_x(xi, bounds, cov_x):
    if cov_x is None:
        return None
    grad = internal2external_grad(xi, bounds)
    grad = np.atleast_2d(grad)
    return np.dot(grad.T, grad) * cov_x


def internal2external(xi, bounds):
    """Convert a series of internal variables to external variables"""

    xe = np.empty_like(xi)

    for i, (v, bound) in enumerate(zip(xi, bounds, strict=False)):
        a = bound[0]  # minimum
        b = bound[1]  # maximum

        if a is None and b is None:  # No constraints
            xe[i] = v

        elif b is None:  # only min
            xe[i] = a - 1.0 + np.sqrt(v**2.0 + 1.0)

        elif a is None:  # only max
            xe[i] = b + 1.0 - np.sqrt(v**2.0 + 1.0)

        else:  # both min and max
            xe[i] = a + ((b - a) / 2.0) * (np.sin(v) + 1.0)

    return xe


def external2internal(xe, bounds):
    """Convert a series of external variables to internal variables"""

    xi = np.empty_like(xe)

    for i, (v, bound) in enumerate(zip(xe, bounds, strict=False)):
        a = bound[0]  # minimum
        b = bound[1]  # maximum

        if a is None and b is None:  # No constraints
            xi[i] = v

        elif b is None:  # only min
            xi[i] = np.sqrt((v - a + 1.0) ** 2.0 - 1)

        elif a is None:  # only max
            xi[i] = np.sqrt((b - v + 1.0) ** 2.0 - 1)

        else:  # both min and max
            xi[i] = np.arcsin((2.0 * (v - a) / (b - a)) - 1.0)

    return xi


def err(p, bounds, efunc, args):

    pe = internal2external(p, bounds)  # convert to external variables
    return efunc(pe, *args)


def calc_cov_x(infodic, p):
    """
    Calculate cov_x from fjac, ipvt and p as is done in leastsq
    """

    fjac = infodic["fjac"]
    ipvt = infodic["ipvt"]
    n = len(p)

    # adapted from leastsq function in scipy/optimize/minpack.py
    perm = np.take(np.eye(n), ipvt - 1, 0)
    r = np.triu(np.transpose(fjac)[:n, :])
    R = np.dot(r, perm)
    try:
        cov_x = np.linalg.inv(np.dot(np.transpose(R), R))
    except LinAlgError:
        cov_x = None
    return cov_x


def leastsqbound(func, x0, bounds, args=(), **kw):
    """
    Constrained multivariant Levenberg-Marquard optimization

    Minimize the sum of squares of a given function using the
    Levenberg-Marquard algorithm. Contraints on parameters are inforced using
    variable transformations as described in the MINUIT User's Guide by
    Fred James and Matthias Winkler.

    Parameters:

    * func      functions to call for optimization.
    * x0        Starting estimate for the minimization.
    * bounds    (min,max) pair for each element of x, defining the bounds on
                that parameter.  Use None for one of min or max when there is
                no bound in that direction.
    * args      Any extra arguments to func are places in this tuple.

    Returns: (x,{cov_x,infodict,mesg},ier)

    Return is described in the scipy.optimize.leastsq function.  x and con_v
    are corrected to take into account the parameter transformation, infodic
    is not corrected.

    Additional keyword arguments are passed directly to the
    scipy.optimize.leastsq algorithm.

    """
    # check for full output
    if kw.get("full_output"):
        full = True
    else:
        full = False

    # convert x0 to internal variables
    i0 = external2internal(x0, bounds)

    # perfrom unconstrained optimization using internal variables
    r = leastsq(err, i0, args=(bounds, func, args), **kw)

    # unpack return convert to external variables and return
    if full:
        xi, cov_xi, infodic, mesg, ier = r
        xe = internal2external(xi, bounds)
        cov_xe = i2e_cov_x(xi, bounds, cov_xi)

        # Correct fjac, ipvt, and qtf to external (bounded) parameter space.
        #
        # The internal-to-external transform is diagonal with gradient
        # ge[i] = d(xe_i)/d(xi_i).  The external covariance cov_xe is already
        # correct.  We rebuild fjac and ipvt from the Cholesky factor of
        # inv(cov_xe) so that calc_cov_x(corrected_infodic, xe) recovers
        # cov_xe exactly.  qtf (= Q^T @ fvec) is corrected to first order via
        # the chain rule for the diagonal transform: qtf_ext ≈ qtf_int / ge.
        if cov_xe is not None:
            ge = internal2external_grad(xi, bounds)
            n = len(xi)
            try:
                # R_ext upper-triangular: R_ext.T @ R_ext = inv(cov_xe)
                R_ext = _cholesky(np.linalg.inv(cov_xe))
                # Store R_ext in scipy's fjac convention: fjac is (n, m) and
                # fjac.T[:n, :] must equal R_ext so that calc_cov_x works.
                fjac_ext = np.zeros_like(infodic["fjac"])
                fjac_ext[:, :n] = R_ext.T
                infodic = dict(infodic)
                infodic["fjac"] = fjac_ext
                infodic["ipvt"] = np.arange(1, n + 1, dtype=np.int32)
                infodic["qtf"] = infodic["qtf"] / ge
            except (LinAlgError, ValueError):
                pass  # Leave infodic unchanged if covariance is singular

        return xe, cov_xe, infodic, mesg, ier

    else:
        xi, ier = r
        xe = internal2external(xi, bounds)
        return xe, ier
