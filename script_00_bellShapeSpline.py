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

font = {"family": "serif",
        "weight": "normal",
        "size": 16}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (9, 6)

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

# %% Interactive plot.
fig, ax = plt.subplots(figsize=(9, 9))
plt.subplots_adjust(bottom=0.3)
ax.set_xlabel("Bell width")
ax.set_ylabel("Bell height")
ax.plot(xy[:, 0], xy[:, 1], "k--", lw=2)
ax.plot(xy[:, 0], xy[:, 1]-t/2., "k-")
ax.plot(xy[:, 0], xy[:, 1]+t/2., "k-")

colours = plt.cm.viridis(np.linspace(0, 1, len(refDeformedShape)))
for i in range(len(refDeformedShape)):
    ax.plot(refDeformedShape[i]["x"], refDeformedShape[i]["y"], "-", c=colours[i])

sldr_ax1 = fig.add_axes([0.15, 0.08, 0.7, 0.05])
sldr1 = matplotlib.widgets.Slider(sldr_ax1, 'Var 1', 0, 1, valinit=0, valfmt="%.1f")
sldr_ax2 = fig.add_axes([0.15, 0.01, 0.7, 0.05])
sldr2 = matplotlib.widgets.Slider(sldr_ax2, 'Var 2', 0, 1, valinit=0, valfmt="%.1f")

lns = None
patches = None
texts = None

def rotatePoint(pt, x0, theta):
    x = pt - x0
    xr = x[0]*np.cos(theta) - x[1]*np.sin(theta)
    yr = x[1]*np.cos(theta) + x[0]*np.sin(theta)
    return np.array([xr, yr]) + x0

def onChanged(val):
    global lns, patches, texts
    if lns is not None:
        for c in lns:
            c.remove()
        for c in patches:
            c.remove()
        for c in texts:
            c.remove()

    cps0 = np.array([
        [0, 0.12],
        [0.35, 0.05],
        [0.97, -0.55],
        [1.0, -1.15],
        [0.92, -0.6],
        [0.3, -0.19],
        [0, -0.12]
    ])

    # Apply 1st rotation
    cps = np.copy(cps0)
    for i in range(1, cps.shape[0]-1):
        cps[i, :] = rotatePoint(cps[i, :], [0, 0], sldr1.val*5./180.*np.pi)

    # Apply 2nd rotation
    x0 = (cps[1, :] + cps[-2, :])/2.
    for i in range(2, cps.shape[0]-2):
        cps[i, :] = rotatePoint(cps[i, :], x0, sldr2.val*20./180.*np.pi)

    # Annotate CP segment lengths.
    texts = []
    for i in range(cps0.shape[0]-1):
        x0 = np.sum(cps[i:i+2, :], axis=0) / 2.
        ds = np.linalg.norm(cps[i+1, :] - cps[i, :])
        texts.append(ax.text(x0[0], x0[1], "{:.3f}".format(ds), va="center", ha="center"))

    # Compute the spline and plot the outline.
    s = np.linspace(0, 1, 101)
    p = np.array([bspline(cps, u, d=2) for u in s])
    lns = ax.plot(cps[:, 0], cps[:, 1], "o--", c="orange", ms=7)
    lns += ax.plot(p[:, 0], p[:, 1], "m-")

    # Fill-in the contour.
    patches = [ax.add_patch(mpatches.Polygon(p[:, :2], facecolor="m", alpha=0.25))]

    # Compute cross-section area.
    area = polyArea(p[:, 0], p[:, 1])
    ax.set_title("Area = {:.4f} units$^2$".format(area))

    return lns

sldr1.on_changed(onChanged)
sldr2.on_changed(onChanged)

lns = onChanged(0.)
