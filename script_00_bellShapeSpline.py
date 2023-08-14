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

font = {"family": "serif",
        "weight": "normal",
        "size": 16}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (9, 6)

# %% Helper function.
def niceFig(xlab=None, ylab=None, figsize=None):
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.tick_params(axis='both', reset=False, which='both', length=5, width=2)
    ax.tick_params(axis='y', direction='out', which="both")
    ax.tick_params(axis='x', direction='out', which="both")
    for spine in ['top', 'right','bottom','left']:
        ax.spines[spine].set_linewidth(2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    return fig, ax

# %% Jellyfish properties.
# Create a function that approximates the shape.
def bellShape(s):
    return np.array([
        np.sign(s)*(1.-np.cos(np.clip(np.abs(s), 0, 1)*np.pi/2.))**0.75,
        np.sin(np.pi/2. - np.clip(np.abs(s), 0, 1)*np.pi/2.)**0.5 - 1.
    ])

# Bell thickness
def bellThickness(s):
    return 0.24 * (np.cos(np.abs(s)*np.pi/2.))**0.65

# %% B-spline
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

# %% Misc

# Implementation of Shoelace formula
def polyArea(x, y):
    return 0.5*np.abs(np.dot(x, np.roll(y, 1) ) - np.dot(y, np.roll(x, 1)))

# Rotate a point around a different point
def rotatePoint(pt, x0, theta):
    x = pt - x0
    xr = x[0]*np.cos(theta) - x[1]*np.sin(theta)
    yr = x[1]*np.cos(theta) + x[0]*np.sin(theta)
    return np.array([xr, yr]) + x0

# %% Fit a b-spline to data of the undeformedshape
n = 501
xy = np.array([bellShape(s) for s in np.linspace(0., 1., n)])
t = np.array([bellThickness(s) for s in np.linspace(0., 1., n)])

# %% Read reference data for the deformed shape.

# From: Bajacaret2009_fig6_bellMotion_cropped
ts = []
refDeformedShape = []
for f in os.listdir("./dataset_01_medusae"):
    if f.startswith("shapeFromAnalytical_"):
        ts = np.append(ts, float(f.split("_")[1].replace(".csv", "")))
        refDeformedShape.append(pandas.read_csv(os.path.join("dataset_01_medusae", f), header=None, names=["x", "y"]))
iSorted = np.argsort(ts)
refDeformedShape = [refDeformedShape[i] for i in iSorted]

# From Costello 2020 fig 9
# Data contain three snapshots of two cycles. Upper and lower part of the medusa
# (aboral and subumbrellar surfaces) for each.
data_Costello = pandas.read_csv("./dataset_01_medusae/deformedShapeHistory_Costello2020_fig9.csv", header=None)
colours = plt.cm.jet(np.linspace(0, 1, 3))
linestyles = ["-", "--"]
fig, ax = plt.subplots()
ax.set_xlabel("Bell width coordinate")
ax.set_ylabel("Bell height coordinate")
for j in range(2):
    for i in range(3):
        i0 = j*4*3+i*4
        # print(j, i, i0)
        # Retrieve the data from the ordered table.
        df = data_Costello[range(i0, i0+4)]
        df.columns = ["xl", "yl", "xu", "yu"]

        # Find the origin of each frame. Assumed at mid-width and top.
        x0 = (df["xu"].max() + df["xu"].min())/2.
        y0 = df["yu"].max()
        # ax.plot(x0, y0, "o", c=colours[i])

        # Plot the data shifted to the origin for each snapshot.
        # ax.plot(df["xl"]-x0+i, df["yl"]-y0, linestyles[j]+"+", c=colours[i])
        # ax.plot(df["xu"]-x0+i, df["yu"]-y0, linestyles[j]+"x", c=colours[i])
        ax.plot(np.abs(df["xl"]-x0)+i/2, df["yl"]-y0, linestyles[j]+"+", alpha=0.25, c=colours[i])
        ax.plot(np.abs(df["xu"]-x0)+i/2, df["yu"]-y0, linestyles[j]+"x", alpha=0.25, c=colours[i])

ylim = ax.get_ylim()
ax.set_ylim(ylim)
ax.set_xlim((-0.1, 1.6))
ax.xaxis.set_ticks(np.arange(0, 1.51, 0.25))

for i in range(4):
    ax.vlines(i/2., ylim[0], ylim[1], color="k", linestyle="dashed")

s = np.linspace(0, 1, 101)

regressedShapes = []

# 1st cycle, upper
cps = np.array([
    [0, 0.],
    [0.05, 0],
    # [0.25, -0.09],
    # [0.46, -0.38],
    [0.472, -0.25],
    [0.389, -0.52]
])
# ax.plot(cps[:, 0], cps[:, 1], "o--", c="orange", ms=7)
pu = np.array([bspline(cps, u, d=2) for u in s])
regressedShapes.append(pu)
ax.plot(pu[:, 0], pu[:, 1], "-", lw=2, c=colours[0])

# 1st cycle lower
cps = np.array([
    [0, -0.2],
    [0.15, -0.2],
    [0.4, -0.35],
    [0.389, -0.52]
])
# ax.plot(cps[:, 0], cps[:, 1], "o--", c="orange", ms=7)
pl = np.array([bspline(cps, u, d=2) for u in s])
regressedShapes.append(pl)
ax.plot(pl[:, 0], pl[:, 1], "-", lw=2, c=colours[0])

xy = np.vstack([pu, np.flipud(pl[:-1, :])])
area = polyArea(xy[:, 0], xy[:, 1])
ax.text(0, 0.05, "A={:.4f} units$^2$".format(area), va="bottom", ha="left")
np.savetxt("dataset_01_medusae/shape_Costello2020_snapshot1.txt", np.hstack([xy, np.zeros((xy.shape[0], 1))]))

# 2nd cycle upper
cps = np.array([
    [0.5, 0.],
    # [0.62, 0.],
    # [0.89, -0.28],
    # [0.89, -0.56]
    [0.62, 0.],
    [0.81, -0.2],
    [0.83, -0.35],
    [0.9, -0.48]
])
# ax.plot(cps[:, 0], cps[:, 1], "o--", c="orange", ms=7)
pu = np.array([bspline(cps, u, d=2) for u in s])
regressedShapes.append(pu)
ax.plot(pu[:, 0], pu[:, 1], "-", lw=2, c=colours[1])

# 2nd cycle lower
cps = np.array([
    [0.5, -0.2],
    # [0.6, -0.2],
    # [0.781, -0.28],
    # [0.89, -0.56]
    [0.66, -0.2],
    [0.81, -0.43],
    [0.9, -0.48]
])
# ax.plot(cps[:, 0], cps[:, 1], "o--", c="orange", ms=7)
pl = np.array([bspline(cps, u, d=2) for u in s])
regressedShapes.append(pl)
ax.plot(pl[:, 0], pl[:, 1], "-", lw=2, c=colours[1])

xy = np.vstack([pu, np.flipud(pl[:-1, :])])
area = polyArea(xy[:, 0], xy[:, 1])
ax.text(0.5, 0.05, "A={:.4f} units$^2$".format(area), va="bottom", ha="left")
np.savetxt("dataset_01_medusae/shape_Costello2020_snapshot2.txt", np.hstack([xy, np.zeros((xy.shape[0], 1))]))

# 3rd cycle upper
cps = np.array([
    [1., 0],
    [1.14, 0.],
    [1.4, -0.34],
    [1.27, -0.66],
])
# ax.plot(cps[:, 0], cps[:, 1], "o--", c="orange", ms=7)
pu = np.array([bspline(cps, u, d=2) for u in s])
regressedShapes.append(pu)
ax.plot(pu[:, 0], pu[:, 1], "-", lw=2, c=colours[2])

# 3rd cycle lower
cps = np.array([
    [1., -0.18],
    [1.12, -0.18],
    [1.3, -0.34],
    [1.27, -0.66],
])
# ax.plot(cps[:, 0], cps[:, 1], "o--", c="orange", ms=7)
pl = np.array([bspline(cps, u, d=2) for u in s])
regressedShapes.append(pl)
ax.plot(pl[:, 0], pl[:, 1], "-", lw=2, c=colours[2])

xy = np.vstack([pu, np.flipud(pl[:-1, :])])
area = polyArea(xy[:, 0], xy[:, 1])
ax.text(1., 0.05, "A={:.4f} units$^2$".format(area), va="bottom", ha="left")
np.savetxt("dataset_01_medusae/shape_Costello2020_snapshot3.txt", np.hstack([xy, np.zeros((xy.shape[0], 1))]))

plt.savefig("./outputs/bellShapeEvolution_Costello2020_data.png", dpi=200, bbox_inches="tight")

# %% Figure out a way to describe each profile using the same set of control points.
# This also has to be done using a single, continuous curve in a way that preserves volume.

def profileFromParams(length, halfThickness, theta, s=np.linspace(0, 1, 101),
                      ax=None, dxPlot=0, colour="b"):
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
            ax.plot([xLast[0]+dxPlot, xNew[0]+dxPlot], [xLast[1], xNew[1]], "ko-", lw=4, alpha=0.25)
            ax.plot(xMid[0]+dxPlot, xMid[1], "rs")
            ax.plot([p0[0]+dxPlot, p1[0]+dxPlot], [p0[1], p1[1]], "r--")

        xLast = xNew

    cps = np.vstack([cps_u, np.flipud(cps_l)])
    xy = np.array([bspline(cps, u, d=2) for u in s])
    area = polyArea(xy[:, 0], xy[:, 1])

    if ax is not None:
        ax.plot(cps[:, 0]+dxPlot, cps[:, 1], "ro")
        ax.plot(xy[:, 0]+dxPlot, xy[:, 1], lw=2, c=colour)
        ax.text(dxPlot, 0.05, "A={:.4f} units$^2$".format(area), va="bottom", ha="left")

    return xy, cps, area


halfThicknesses = np.array([
    [0.101, 0.099, 0.090],
    [0.081, 0.102, 0.077],
    [0.063, 0.069, 0.048],
# [0.033, 0.031, 0.024],
    [0.033, 0.041, 0.0338],
    [0.011, 0.011, 0.011],
])

lengths = np.array([
    [0.080, 0.052, 0.103],
    [0.118, 0.105, 0.163],
    [0.194, 0.250, 0.194],
    [0.228, 0.179, 0.228],
    [0.250, 0.165, 0.250],
])

thetas = np.array([
    [0., 0., 0.],
# [0.440, 0.147, 0.427],
# [0.462, 0.831, 1.279],
# [1.102, 1.570, 1.437],
# [2.200, 0.125, 2.200],
    [0.380, 0.080, 0.250],
    [0.400, 0.731, 1.100],
    [0.900, 1.670, 1.370],
    [2.200, 0.125, 2.100],
])

# Idea 1 - define a central "backbone" with control points offset by a normal distance
# from it. Moving the backbone will automatically move the control points. Adjusting the
# normal distance will allow fine-tuning of contained volume to ensure mass continuity.

fig, ax = plt.subplots()
ax.set_xlabel("Bell width coordinate")
ax.set_ylabel("Bell height coordinate")

# Plot the shapes.
for i in range(2):
    for j in range(3):
        ax.plot(regressedShapes[j*2+i][:, 0], regressedShapes[j*2+i][:, 1], "k-", alpha=0.5)

# Set up axes.
ylim = ax.get_ylim()
ax.set_ylim(ylim)
ax.set_xlim((-0.1, 1.6))
ax.xaxis.set_ticks(np.arange(0, 1.51, 0.25))
for i in range(4):
    ax.vlines(i/2., ylim[0], ylim[1], color="k", linestyle="dashed")

# Create the backbone and control points.
for j in range(3):
    xy, cps, area = profileFromParams(lengths[:, j], halfThicknesses[:, j], thetas[:, j],
                                      s=np.linspace(0, 1, 101), ax=ax, dxPlot=0.5*j, colour=colours[j])

plt.savefig("./outputs/bellShapeEvolution_Costello2020_kinematicsModel.png", dpi=200, bbox_inches="tight")

# %% Create time series of the bell shape parameters for continuous motion.

# Function for smoothing the sparse kinematics data. A lot of fidgeting involved
# to make sure the curves are continuous when going from t=1 to t=0 and to avoid
# substantial overshooting.
def smoothWithSpline(tf, yf, t):
    tf = [tf[0], tf[0]+0.01, tf[1], tf[2], tf[-1]-0.5, tf[-1]-0.4, tf[-1]-0.15, tf[-1]]
    yf = [yf[0], yf[0], yf[1], yf[2], yf[-1], yf[-1], yf[-1], yf[-1]]
    spl = scipy.interpolate.splrep(tf, yf, s=0., k=2)
    return scipy.interpolate.splev(t, spl)

# TODO digitise from Costello plot.
timeVals = [0, 0.2, 0.4, 1.0]

fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(8, 12))
axes[-1].set_xlabel("t/T")
axes[0].set_ylabel("Length")
axes[1].set_ylabel("Half-thickness")
axes[2].set_ylabel("Angle")
for ax in axes:
    ax.xaxis.set_ticks(timeVals)
plt.subplots_adjust(top=0.985, bottom=0.13, left=0.14, right=0.97, hspace=0.15, wspace=0.195)

L = np.hstack([lengths, lengths[:, 0][:, np.newaxis]])
thk = np.hstack([halfThicknesses, halfThicknesses[:, 0][:, np.newaxis]])
theta = np.hstack([thetas, thetas[:, 0][:, np.newaxis]])
colours = plt.cm.Greens(np.linspace(0.3, 1, L.shape[0]))
for i in range(L.shape[0]):
    t = np.linspace(0, 1, 101)
    Lfit = smoothWithSpline(timeVals, L[i, :], t)
    thkFit = smoothWithSpline(timeVals, thk[i, :], t)
    thetaFit = smoothWithSpline(timeVals, theta[i, :], t)

    axes[0].plot(timeVals, L[i, :]+i*0, "o--", c=colours[i], alpha=1)
    axes[0].plot(t, Lfit, c=colours[i], lw=2)
    axes[1].plot(timeVals, thk[i, :]+i*0, "o--", c=colours[i], alpha=1)
    axes[1].plot(t, thkFit, c=colours[i], lw=2)
    axes[2].plot(timeVals, theta[i, :]+i*0, "o--", c=colours[i], alpha=1)
    axes[2].plot(t, thetaFit, c=colours[i], lw=2)

plt.savefig("./outputs/bellShapeEvolution_Costello2020_kinematicsParams.png", dpi=200, bbox_inches="tight")

# %% Test applying volume conservation

def thicknessTarget(x, Lfit, thkFit, thetaFit):
    dt = (x-0.5)/0.5*0.01
    xy1, cps1, area1 = profileFromParams(Lfit, thkFit+dt, thetaFit)
    return abs(area1 - 0.0644)

def NewtRaph(func, x0, funcKwargs={}, Nmax=100, tol=5e-18):
    """ Newton-Rhapson method for root-finding in 1D
    Arguments
    ---------
        @parm func - pointer to a function which accepts x and returns y(x) and dy/dx(x)
        @param x0 - initial guess of the root
        @param funcKwargs - dictionary of optional keywords used by the function
        @param Nmax - max no. iterations
        @param tol - tolerance (absolute!) down to which y(x)=0 is computed
    Returns
    --------
        @param x - value of x where y(x) == 0
    """
    x = x0
    for i in range(Nmax):
        f, df = func(x, **funcKwargs)
        dx = f/df
        x -= dx
        print(i, f, df, dx)
        if np.abs(dx) < tol:
            return x
    raise RuntimeError("Did not converge within {:d} iterations".format(Nmax))

# dt = NewtRaph(thicknessTarget, 0.01, funcKwargs={"Lfit": Lfit, "thkFit": thkFit, "thetaFit": thetaFit})

res = scipy.optimize.minimize(
    thicknessTarget, [0.],
    args=(Lfit, thkFit, thetaFit),
    bounds=[(0, 1)], constraints=(),
    tol=1e-12, callback=None, options={"disp": False}, method="SLSQP")

# %% Animate the jellyfish!

# Set up the plot
fig, ax = niceFig(figsize=(8, 10))
ax.set_ylabel("Bell height coordinate")
ax.set_xlabel("Bell width coordinate")
ax.set_xlim((0, 0.6))
ax.set_ylim((-0.8, 0.2))
ax.set_aspect("equal", "box")

# TODO Plot constant elements

class JellyfishPlot(object):
    def __init__(self, T):#, arr_c, arr_v):
        # self.cs = cs
        self.T = T
        self.plotObjects = None
        # self.arr_c = arr_c
        # self.arr_v = arr_v

    def animate(self, i):
        # for c in self.cs.collections:
        #     c.remove()
        if self.plotObjects is not None:
            for c in self.plotObjects:
                c.remove()

        # Compute the point in the period.
        t = i/self.T % 1

        # Get parameters for each control point.
        Lfit, thkFit, thetaFit = [], [], []
        for j in range(L.shape[0]):
            Lfit.append(smoothWithSpline(timeVals, L[j, :], t))
            thkFit.append(smoothWithSpline(timeVals, thk[j, :], t))
            thetaFit.append(smoothWithSpline(timeVals, theta[j, :], t))

        # Correct thickness to achieve target cross-section area.
        res = scipy.optimize.minimize(
            thicknessTarget, [0.],
            args=(Lfit, thkFit, thetaFit),
            bounds=[(0, 1)], constraints=(),
            tol=1e-12, callback=None, options={"disp": False}, method="SLSQP")
        thkFit += (res.x[0]-0.5)/0.5*0.01

        # Compute the profile.
        xy, cps, area = profileFromParams(Lfit, thkFit, thetaFit)

        # Plot.
        self.plotObjects = ax.plot(xy[:, 0], xy[:, 1], c="r", lw=2)
        self.plotObjects += ax.plot(cps[:, 0], cps[:, 1], "ko--", lw=1, alpha=0.25)
        ax.set_title("A={:.4f} units$^2$".format(area))

        return self.plotObjects

jellyfishPlot = JellyfishPlot(150)

anim = animation.FuncAnimation(fig, jellyfishPlot.animate, repeat=False, frames=range(150))
writer = animation.PillowWriter(fps=20)
anim.save('./outputs/jellyfishAnimation.gif', writer=writer, dpi=200)

# %% Interactive plot.
# print(a)
# fig, ax = plt.subplots(figsize=(9, 9))
# plt.subplots_adjust(bottom=0.3)
# ax.set_xlabel("Bell width")
# ax.set_ylabel("Bell height")
# ax.plot(xy[:, 0], xy[:, 1], "k--", lw=2)
# ax.plot(xy[:, 0], xy[:, 1]-t/2., "k-")
# ax.plot(xy[:, 0], xy[:, 1]+t/2., "k-")

# colours = plt.cm.viridis(np.linspace(0, 1, len(refDeformedShape)))
# for i in range(len(refDeformedShape)):
#     ax.plot(refDeformedShape[i]["x"], refDeformedShape[i]["y"], "-", c=colours[i])

# sldr_ax1 = fig.add_axes([0.15, 0.15, 0.7, 0.05])
# sldr1 = matplotlib.widgets.Slider(sldr_ax1, 'Var 1', 0, 1, valinit=0, valfmt="%.1f")
# sldr_ax2 = fig.add_axes([0.15, 0.08, 0.7, 0.05])
# sldr2 = matplotlib.widgets.Slider(sldr_ax2, 'Var 2', 0, 1, valinit=0, valfmt="%.1f")
# sldr_ax3 = fig.add_axes([0.15, 0.01, 0.7, 0.05])
# sldr3 = matplotlib.widgets.Slider(sldr_ax3, 'Var 3', 0, 1, valinit=0, valfmt="%.1f")

# lns = None
# patches = None
# texts = None

# def onChanged(val):
#     global lns, patches, texts
#     if lns is not None:
#         for c in lns:
#             c.remove()
#         for c in patches:
#             c.remove()
#         for c in texts:
#             c.remove()

#     # ---

#     # Version 1 - rotate points of the bounding spline.
#     # => does not preserve volume.
#     cps0 = np.array([
#         [0, 0.12],
#         [0.35, 0.05],
#         [0.97, -0.55],
#         [1.0, -1.15],
#         [0.92, -0.6],
#         [0.3, -0.19],
#         [0, -0.12]
#     ])

#     # Apply 1st rotation
#     cps = np.copy(cps0)
#     cps[1, :] = rotatePoint(cps[1, :], cps[0, :], sldr1.val*5./180.*np.pi)
#     cps[-2, :] = rotatePoint(cps[-2, :], cps[-1, :], sldr1.val*5./180.*np.pi)
#     for i in range(2, cps.shape[0]-2):
#         cps[i, :] = rotatePoint(cps[i, :], [0, 0], sldr1.val*5./180.*np.pi)

#     # Apply 2nd rotation
#     cps[2, :] = rotatePoint(cps[2, :], cps[1, :], sldr2.val*20./180.*np.pi)
#     cps[-3, :] = rotatePoint(cps[-3, :], cps[-2, :], sldr2.val*20./180.*np.pi)
#     x0 = (cps[1, :] + cps[-2, :])/2.
#     for i in range(3, cps.shape[0]-3):
#         cps[i, :] = rotatePoint(cps[i, :], x0, sldr2.val*20./180.*np.pi)

#     # Rotate the tip.
#     x0 = (cps[2, :] + cps[4, :])/2.
#     cps[3, :] = rotatePoint(cps[3, :], x0, sldr3.val*90./180.*np.pi)

#     # Version 2 - work with a "backbone" spline and derive CP positions from that
#     #   based on local thickness and normal vectors. Scale the thickness iteratively
#     #   to enforce volume conservation to a particular threshold.
#     # =>

#     # TODO

#     # ---

#     # Annotate CP segment lengths.
#     texts = []
#     for i in range(cps0.shape[0]-1):
#         x0 = np.sum(cps[i:i+2, :], axis=0) / 2.
#         ds = np.linalg.norm(cps[i+1, :] - cps[i, :])
#         texts.append(ax.text(x0[0], x0[1], "{:.3f}".format(ds), va="center", ha="center"))

#     # Compute the spline and plot the outline.
#     s = np.linspace(0, 1, 101)
#     p = np.array([bspline(cps, u, d=2) for u in s])
#     lns = ax.plot(cps[:, 0], cps[:, 1], "o--", c="orange", ms=7)
#     lns += ax.plot(p[:, 0], p[:, 1], "m-")

#     # Fill-in the contour.
#     patches = [ax.add_patch(mpatches.Polygon(p[:, :2], facecolor="m", alpha=0.25))]

#     # Compute cross-section area.
#     area = polyArea(p[:, 0], p[:, 1])
#     ax.set_title("Area = {:.4f} units$^2$".format(area))

#     return lns

# sldr1.on_changed(onChanged)
# sldr2.on_changed(onChanged)
# sldr3.on_changed(onChanged)

# lns = onChanged(0.)
