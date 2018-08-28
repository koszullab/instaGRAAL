#!/usr/bin/env python3

import numpy as np
from scipy.optimize import leastsq
from scipy.optimize import fsolve

d = 2


def residuals(p, y, x):

    kuhn, lm, slope, A = p
    # d = 1.5
    # d = 5.0
    rippe = A * (
        0.53
        * (kuhn ** -3.)
        * np.power((lm * x / kuhn), slope)
        * np.exp((d - 2) / ((np.power((lm * x / kuhn), 2) + d)))
    )
    error = y - rippe
    return error


def peval(x, param):

    # d = 1.5
    # d = 5.0
    rippe = param[3] * (
        0.53
        * (param[0] ** -3.)
        * np.power((param[1] * x / param[0]), (param[2]))
        * np.exp((d - 2) / ((np.power((param[1] * x / param[0]), 2) + d)))
    )
    return rippe


def log_residuals(p, y, x):
    kuhn, lm, slope, A = p
    # d = 1.5
    # d = 5.0
    rippe = (
        np.log(A)
        + np.log(0.53)
        - 3 * np.log(kuhn)
        + slope * (np.log(lm * x / kuhn))
        + (d - 2) / ((np.power((lm * x / kuhn), 2) + d))
    )
    error = y - rippe

    return error


def log_peval(x, param):
    # d = 1.5
    # d = 5.0
    rippe = (
        np.log(param[3])
        + np.log(0.53)
        - 3 * np.log(param[0])
        + param[2] * (np.log(param[1] * x) - np.log(param[0]))
        + (d - 2) / ((np.power((param[1] * x / param[0]), 2) + d))
    )
    return rippe


def estimate_param_rippe(y_meas, x_bins):
    # kuhn = 2000
    kuhn = 50
    # kuhn = 500
    # lm = 200
    lm = 9.6  # ok s1 tricho
    slope = -1.5
    # d = 3
    # d = 1.5
    # d = 5.0
    # A = np.sum(y_meas)
    # A = np.sum(y_meas) #s1
    # A = np.max(y_meas) * 0.05
    A = np.max(y_meas)
    p0 = [kuhn, lm, slope, A]
    lower_fact = 7.
    plsq = leastsq(
        log_residuals, p0, args=(np.log(y_meas / lower_fact), x_bins)
    )
    # bounds = []
    # bounds.append((0., 500.))
    # bounds.append((9.0, 9.6))
    # bounds.append((-1., 0.))
    # bounds.append((0,  A * 2))
    # plsq = leastsqbound(log_residuals, p0,
    # bounds=bounds,args=(np.log(y_meas), x_bins))
    print(plsq)

    # plsq[0][4] = plsq[0][4]
    y_estim = peval(x_bins, plsq[0])
    kuhn_x, lm_x, slope_x, A_x = plsq[0]
    plsq_out = [kuhn_x, lm_x, slope_x, d, A_x]

    np_plsq = np.array(plsq_out)
    # print "parameters from optimization = ", np_plsq
    if np.any(np.isnan(np_plsq)) or slope_x >= 0:
        print("warning : pb in parameters estimation")
        A = np.max(y_meas)
        test = peval(x_bins, [kuhn, lm, slope, A])
        new_A = y_meas[0] * A / test.max()
        plsq_out = [kuhn, lm, slope, d, A * new_A]
        y_estim = peval(x_bins, [kuhn, lm, slope, new_A])
        print("y estim = ", y_estim)

    return plsq_out, y_estim


def residual_4_max_dist(x, p):

    kuhn, lm, slope, d, A, y = p
    x[np.isnan(x)] = 0
    x = np.abs(x)

    rippe = A * (
        0.53
        * (kuhn ** -3.)
        * np.power((lm * x / kuhn), slope)
        * np.exp((d - 2) / ((np.power((lm * x / kuhn), 2) + d)))
    )
    error = y - rippe
    return np.abs(error)


def estimate_max_dist_intra(p, val_inter):
    s0 = 500
    # print "estimate max distance trans = ",p
    kuhn, lm, slope, d, A = p
    p0 = [kuhn, lm, slope, d, A, val_inter]
    x = fsolve(residual_4_max_dist, s0, args=(p0))
    # print "limit inter/intra distance = ", x
    # print "val trans = ", peval(x, p)
    # raw_input("alors?")
    # print "val_inter  = ",val_inter
    solution = x[0]

    return np.abs(solution)


def estimate_max_dist_intra_nuis(p, val_inter, old_s):
    s0 = old_s
    # print "estimate max distance trans = ",p
    kuhn, lm, slope, d, A = p
    p0 = [kuhn, lm, slope, d, A, val_inter]
    x = fsolve(residual_4_max_dist, s0, args=(p0))
    # print "limit inter/intra distance = ", x
    # print "val trans = ", peval(x, p)
    # raw_input("alors?")
    # print "val_inter  = ",val_inter
    return x[0]
