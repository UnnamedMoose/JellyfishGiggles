# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 10:04:05 2023

@author: ALidtke
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import os
import re
import platform
import matplotlib.patches as mpatches
import scipy.interpolate
import matplotlib.animation as animation
import scipy.optimize

# %% B-splines
def CoxDeBoor(knots, u, k, d, count):
    if (d == 0):
        return int((knots[k] <= u and u < knots[k+1]) or ((u >= (1.0-1e-12)) and (k == (count-1))))
    return ((u-knots[k]) / max(1e-12, knots[k+d] - knots[k])) * CoxDeBoor(knots, u, k, (d-1), count) \
        + ((knots[k+d+1]-u) / max(1e-12, knots[k+d+1] - knots[k+1])) * CoxDeBoor(knots, u, (k+1), (d-1), count)

def bspline(cv, s, d=3):
    count = len(cv)
    knots = np.array([0]*d + list(range(count-d+1)) + [count-d]*d, dtype='float') / (count-d)
    pt = np.zeros(cv.shape[1])
    for k in range(count):
        pt += CoxDeBoor(knots, s, k, d, count) * cv[k]
    return pt

# Implementation of Shoelace formula
def polyArea(x, y):
    return 0.5*np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

# Rotate a point around a different point
def rotatePoint(pt, x0, theta):
    x = pt - x0
    xr = x[0]*np.cos(theta) - x[1]*np.sin(theta)
    yr = x[1]*np.cos(theta) + x[0]*np.sin(theta)
    return np.array([xr, yr]) + x0

# Evaluation of smoothing splines. Alternative for scipy.interp
def getSegmentPosition(iSeg, tTarget, cps, NiterMax=100, tol=1e-6, printout=False):
    # For an arbitrary t/T value, need to either pre-compute the splines describing
    # the motion of each segment at the right time values or have an iterative routine
    # for matching the desired time.
    s0 = 0.
    s1 = 1.
    y = 0.

    for i in range(NiterMax):
        # Evaluate at the centre of the trust region.
        sx = (s0 + s1) / 2.

        # Pick control points from the array and make a spline
        cps_y = cps[[0, iSeg+1], :]
        #t, y = old_evaluate_spline(cps_y, [sx])
        t, y = bspline(cps_y.T, sx)

        if printout:
            print(i, " ", abs(t-tTarget), " ", t)

        # Check if converge. Shrink the trust region if not.
        if abs(t-tTarget) < tol:
            return y
        if t > tTarget:
            s1 = sx
        else:
            s0 = sx

    # Return the best available approximation even though convergence has not been met.
    return y

# Generation of motion.
def profileFromParams(length, halfThickness, theta, s=np.linspace(0, 1, 101),
                      ax=None, dxPlot=0, scalePlot=1., colour="b"):
    """
    Creates a jellyfish profile from a set of segment lengths, thicknesses and rotation angles.
    Can also provide additional parameters to plot the profile on an external figure.

    Parameters
    ----------
    length : TYPE
        DESCRIPTION.
    halfThickness : TYPE
        DESCRIPTION.
    theta : TYPE
        DESCRIPTION.
    s : TYPE, optional
        DESCRIPTION. The default is np.linspace(0, 1, 101).
    ax : TYPE, optional
        DESCRIPTION. The default is None.
    dxPlot : TYPE, optional
        DESCRIPTION. The default is 0.
    colour : TYPE, optional
        DESCRIPTION. The default is "b".

    Returns
    -------
    xy : TYPE
        DESCRIPTION.
    cps : TYPE
        DESCRIPTION.
    area : TYPE
        DESCRIPTION.

    """
    cps_u, cps_l = [], []

    xLast = np.array([-length[0]/2., -halfThickness[0]])
    for i in range(len(halfThickness)):
        xNew = xLast + [length[i], 0.]
        xNew = rotatePoint(xNew, xLast, -theta[i])
        xMid = (xLast + xNew) / 2.
        vTan = (xNew - xLast) / np.linalg.norm(xNew - xLast)
        vPer = np.array([-vTan[1], vTan[0]])

        p0 = xMid + halfThickness[i]*vPer
        p1 = xMid - halfThickness[i]*vPer
        cps_u.append(p0)
        cps_l.append(p1)

        if ax is not None:
            ax.plot([xLast[0]*scalePlot+dxPlot*scalePlot, xNew[0]*scalePlot+dxPlot*scalePlot],
                    [xLast[1]*scalePlot, xNew[1]*scalePlot], "ko-", lw=4, alpha=0.25)
            ax.plot(xMid[0]*scalePlot+dxPlot*scalePlot, xMid[1]*scalePlot, "rs")
            ax.plot([p0[0]*scalePlot+dxPlot*scalePlot, p1[0]*scalePlot+dxPlot*scalePlot],
                    [p0[1]*scalePlot, p1[1]*scalePlot], "r--")

        xLast = xNew

    cps = np.vstack([cps_u, np.flipud(cps_l)])
    xy = np.array([bspline(cps, u, d=2) for u in s])
    area = polyArea(xy[:, 0], xy[:, 1])

    if ax is not None:
        ax.plot(cps[:, 0]*scalePlot+dxPlot*scalePlot, cps[:, 1]*scalePlot, "ro")
        ax.plot(xy[:, 0]*scalePlot+dxPlot*scalePlot, xy[:, 1]*scalePlot, lw=2, c=colour)
        ax.text(dxPlot*scalePlot, 0.05*scalePlot, "A={:.4f} units$^2$".format(area*scalePlot**2.),
                va="bottom", ha="left")

    return xy, cps, area
