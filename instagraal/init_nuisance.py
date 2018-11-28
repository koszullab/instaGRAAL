#!/usr/bin/env python3

from scipy.optimize import minimize
import numpy as np
from scipy.optimize import fmin_slsqp
from scipy.optimize import fsolve

# from scipy.optimize import minimize
from scipy.optimize import leastsq
from instagraal.leastsqbound import *

d0 = 1.0  # distance bias Hi-C
d_exp = -10.0
import matplotlib.pyplot as plt


def log_residuals_4_min(param, y, x):

    d_init, alpha_0, alpha_1, A = param
    hic_c = np.zeros(x.shape)
    log_val_lim_0 = (
        np.log(A)
        + (alpha_0 - alpha_1) * np.log(d_init)
        + ((d_exp - 2) / (np.power(d_init, 2) + d_exp))
    )
    for i in range(0, len(hic_c)):
        if x[i] < d_init and x[i] > 0:
            hic_c[i] = (
                np.log(A)
                + np.log(x[i]) * alpha_0
                + ((d_exp - 2) / (np.power(x[i], 2) + d_exp))
            )
        else:
            hic_c[i] = log_val_lim_0 + np.log(x[i]) * alpha_1

    err = np.sqrt(np.power(y - hic_c, 2).sum())

    return err


def log_residuals(param, y, x):

    alpha_0, alpha_1, A = param
    hic_c = np.zeros(x.shape)
    log_val_lim_0 = (
        np.log(A)
        + (alpha_0 - alpha_1) * np.log(d0)
        + ((d_exp - 2) / (np.power(d0, 2) + d_exp))
    )
    for i in range(0, len(hic_c)):
        if x[i] <= 0:
            hic_c[i] = 0
        elif x[i] < d0 and x[i] > 0:
            hic_c[i] = (
                np.log(A)
                + np.log(x[i]) * alpha_0
                + ((d_exp - 2) / (np.power(x[i], 2) + d_exp))
            )
        else:
            hic_c[i] = log_val_lim_0 + np.log(x[i]) * alpha_1

    err = y - hic_c

    return err


def residuals(param, y, x):

    alpha_0, alpha_1, A = param
    hic_c = np.zeros(x.shape)
    val_lim_0 = A * np.power(d0, alpha_0 - alpha_1)
    for i in range(0, len(hic_c)):
        if x[i] < d0:
            hic_c[i] = A * np.power(x[i], alpha_0)
        else:
            hic_c[i] = val_lim_0 * np.power(x[i], alpha_1)

    err = y - hic_c

    return err


def peval(x, param):

    d_init, alpha_0, alpha_1, A = param
    hic_c = np.zeros(x.shape)
    val_lim_0 = (
        A
        * np.power(d_init, alpha_0 - alpha_1)
        * np.exp((d_exp - 2) / (np.power(d_init, 2) + d_exp))
    )
    for i in range(0, len(hic_c)):
        if x[i] < d_init:
            hic_c[i] = (
                A
                * np.power(x[i], alpha_0)
                * np.exp((d_exp - 2) / (np.power(x[i], 2) + d_exp))
            )
        else:
            hic_c[i] = val_lim_0 * np.power(x[i], alpha_1)

    return hic_c


def estimate_param_hic(y_meas, x_bins):
    alpha_0 = -10
    alpha_1 = -1.5
    x0 = x_bins.min()
    print("x0 = ", x0)
    A = (
        y_meas.max()
        * (x0 ** (-alpha_0))
        / np.exp((d_exp - 2) / (x0 ** 2 + d_exp))
    )
    print("A = ", A)
    p0 = [alpha_0, alpha_1, A]
    args = (np.log(y_meas), x_bins)
    plsq = leastsq(log_residuals, p0, args=args)

    plsq[0]
    print(plsq)

    bnds = ((0, 3), (-10, -0.2), (-2, -0.2), (0, None))
    alpha_0, alpha_1, A = plsq[0]
    p0 = [d0, alpha_0, alpha_1, A]
    # cns = ({'type': 'ineq', 'fun': lambda x:  x[0] - x[1]})
    # alpha_0 > alpha_1
    cns = {"type": "ineq", "fun": lambda x: x[1] - x[0]}

    res = minimize(
        log_residuals_4_min,
        p0,
        args=args,
        method="L-BFGS-B",
        bounds=bnds,
        constraints=cns,
        options={"disp": True},
    )
    print("res = ", res)
    y_estim_sls = peval(x_bins, res.x)

    plt.loglog(x_bins, y_estim_sls)
    plt.loglog(x_bins, y_meas)
    plt.show()
    return res, y_estim_sls


def residual_4_max_dist(x, p):
    d_init, alpha_0, alpha_1, A, y = p
    hic_c = np.zeros(x.shape)
    val_lim_0 = (
        A
        * np.power(d_init, alpha_0 - alpha_1)
        * np.exp((d_exp - 2) / (np.power(d_init, 2) + d_exp))
    )
    for i in range(0, len(hic_c)):
        if x[i] < d_init:
            hic_c[i] = (
                A
                * np.power(x[i], alpha_0)
                * np.exp((d_exp - 2) / (np.power(x[i], 2) + d_exp))
            )
        else:
            hic_c[i] = val_lim_0 * np.power(x[i], alpha_1)
    err = y - hic_c
    return err


def estimate_max_dist_intra(p, val_inter):
    print("val_inter = ", val_inter)

    d_init, alpha_0, alpha_1, A = p
    p0 = [d_init, alpha_0, alpha_1, A, val_inter]
    s0 = 500
    x = fsolve(residual_4_max_dist, s0, args=(p0))
    print("limit inter/intra distance = ", x)
    print("val model @ dist inter = ", peval(x, p))
    return x[0]
