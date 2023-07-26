using Plots
using Images
using LinearAlgebra
using StaticArrays
using Interpolations
using BasicBSpline

# Create a function that approximates the shape.
bellShapeX = s -> sign(s)*(1.0f0-cos(clamp(abs(s), 0, 1)*pi/2.0f0))^0.75f0
bellShapeY = s -> (sin(pi/2.0f0 - clamp(abs(s), 0, 1)*(pi/2.0f0)))^0.5f0 - 1.0f0

# Redo with a spline.
# T = np.hstack([[0]*M, np.arange(pts.shape[0]-M+1), [pts.shape[0]-M]*M])
# p = 3
# kVec = Float64[0, 0, 0, 0, 1, 2, 2, 2, 2]
# controlPoints = [[0, 0], [0.5, 0.0], [1, 0], [1, -0.5], [1, -1]]
# p = 2
# kVec = Float64[0, 0, 0, 1, 2, 2, 2]
# controlPoints = [[0, 0], [0.2, 0], [0.97, -0.55], [1, -1]]
p = 3
kVec = Float64[0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4]
controlPoints = [[-1, -1], [-0.97, -0.55], [-0.2, 0], [0, 0], [0.2, 0], [0.97, -0.55], [1, -1]]

kVec ./= maximum(kVec)
k = KnotVector(kVec)
P = BSplineSpace{p}(k)
bell_spline = BSplineManifold(controlPoints, P)

# Evaluate for plotting.
xy = bell_spline.(0:0.01:1)
x = [v[1] for v in xy]
y = [v[2] for v in xy]
xy_spline = hcat(x, y)'
xc = [v[1] for v in controlPoints]
yc = [v[2] for v in controlPoints]
xy_cp = hcat(xc, yc)'

# Create the shape (non-dimensional) with origin at the bell top
s = -1:0.01:1
x_bell = bellShapeX.(s)
y_bell = bellShapeY.(s)
plot(x_bell, y_bell, lw=2, c=:blue, label="Analytical",
    xlabel="Bell width coord", ylabel="Bell height coord")
plot!(xy_spline[1, :], xy_spline[2, :], c=:orange, lw=2, label="B-spline")
plot!(xy_cp[1, :], xy_cp[2, :], c=:black, ls=:dash, marker=:circle, ms=5, label=nothing)

savefig("outputs/plot_00_bellShape.png")

