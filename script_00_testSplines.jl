using Plots
using Images
using LinearAlgebra
using StaticArrays
using Interpolations
using ColorSchemes
using Optim

using ParametricBodies
#include("D:/git/ParametricBodies.jl/src/NurbsCurves.jl")

include("./src/splines.jl")
#using .JellyfishPhysics

# ===
# Test splines
s = 0:0.01:1

# 1st cycle, upper
cps_u = hcat(
    [0, 0.],
    [0.05, 0],
    [0.472, -0.25],
    [0.389, -0.52]
)

pu = xold_evaluate_spline(cps_u, s)

# 1st cycle lower
cps_l = hcat(
    [0, -0.2],
    [0.15, -0.2],
    [0.4, -0.35],
    [0.389, -0.52]
)

pl = xold_evaluate_spline(cps_l, s)
xy = hcat(pu, reverse(pl[:, 1:end-1], dims=2))

# Plot own spline
plot(pu[1, :], pu[2, :], label="", color=:blue, show=true)
plot!(cps_u[1, :], cps_u[2, :], linestyle=:dash, marker=:circle, label="", color=:blue)
plot!(pl[1, :], pl[2, :], label="", color=:red)
plot!(cps_l[1, :], cps_l[2, :], linestyle=:dash, marker=:circle, label="", color=:red)

# Plot PB spline
cps = SArray{Tuple{2, 4}}(cps_l)
cps_m = MMatrix(cps)
# weights = SA[1.,1.,1.]
# knots =   SA[0,0,0,1,1,1.]
# make a nurbs curve
# circle = NurbsCurve(cps_m,knots,weights)
curve = BSplineCurve(cps_m; degree=2)
xy = hcat([curve(u, 0) for u in s]...)
plot!(xy[1, :], xy[2, :], color=:black, lw=2, linestyle=:dash)

cps = SArray{Tuple{2, 4}}(cps_u)
cps_m = MMatrix(cps)
curve = BSplineCurve(cps_m; degree=2)
xy = hcat([curve(u, 0) for u in s]...)
plot!(xy[1, :], xy[2, :], color=:black, lw=2, linestyle=:dash)
plot!(title="Comparison of own splines and ParametricBodies")
savefig("outputs/plot_00_testSplines_ownVsParametricBodies.png")

# ===
# Test rotation
cps = hcat(
    [0, -0.2],
    [0.15, -0.2],
    [0.4, -0.35],
    [0.389, -0.52],
)
pl = pbSpline(cps, s)
plot(pl[1, :], pl[2, :], label="", color=:red, show=true)
plot!(cps[1, :], cps[2, :], linestyle=:dash, marker=:circle, label="", color=:red)

cps = hcat(
    rotatePoint([0, -0.2], [0., -0.2], 15.0/180.0*pi),
    rotatePoint([0.15, -0.2], [0., -0.2], 15.0/180.0*pi),
    rotatePoint([0.4, -0.35], [0., -0.2], 15.0/180.0*pi),
    rotatePoint([0.389, -0.52], [0., -0.2], 15.0/180.0*pi),
)
pl = pbSpline(cps, s)
plot!(pl[1, :], pl[2, :], label="", color=:orange)
plot!(cps[1, :], cps[2, :], linestyle=:dash, marker=:circle, label="", color=:orange)

savefig("outputs/plot_00_testSplines_testSplinesAndRotation.png")
