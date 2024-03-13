using Plots
using Images
using LinearAlgebra
using StaticArrays
using Interpolations
using ColorSchemes
using Optim

using ParametricBodies

#include("./src/JellyfishPhysics.jl")
#using .JellyfishPhysics

include("./src/splines.jl")
# xy = pbSpline(cps, s)

# ===
# Test splines
s = 0:0.01:1

# ===
# Bell shape parameters - each row is for a separate segment, three snapshots measured in total.
thetas = hcat(
    [0., 0., 0.],
    [0., 0., 0.],
    [0.390, 0.11, 0.37],
    [0.400, 0.731, 0.920],
    [0.950, 1.75, 1.44],
    [2.200, 0.125, 2.100],
)'

lengths = hcat(
    [0.020, 0.013, 0.025],
    [0.040, 0.026, 0.0515],
    [0.12, 0.11, 0.14],
    [0.194, 0.250, 0.194],
    [0.228, 0.179, 0.228],
    [0.250, 0.165, 0.250],
)'

halfThicknesses = hcat(
    [0.097, 0.099, 0.090],
    [0.097, 0.099, 0.090],
    [0.083, 0.102, 0.080],
    [0.063, 0.071, 0.049],
    [0.0335, 0.042, 0.0345],
    [0.011, 0.011, 0.011],
)'

# Digitised from Costello plot.
#timeVals = [0, 0.01, 0.2, 0.4, 0.95, 1.0]
# timeVals = [0.04836558, 0.22848566, 0.41027352, 1.39926618, 1.57771848, 1.76617745];
timeVals = [0, 0.13, 0.27]

# ===
# Regression using splines.
tSplineCps = [0, 0.025, 0.15, 0.25, 0.45, 0.9, 1.0]

cps_thetas = hcat([
    [0., 0.,        0., 0., 0.,       0., 0.],
    [0., 0.,        0., 0., 0.,       0., 0.],
    [0.39, 0.39,        0., 0.38, 0.39,       0.39, 0.39],
    [0.4, 0.4,        0.75, 1.0, 0.6,       0.4, 0.4],
    [0.95, 0.95,        1.95, 1.4, 1.1,       0.95, 0.95],
    [2.2, 2.2,        -0.5, 2.1, 2.15,       2.2, 2.2],
]...)'

cps_lengths = hcat([
    [0.02, 0.02,        0.01, 0.025, 0.024,       0.02, 0.02],
    [0.04, 0.04,        0.02, 0.052, 0.049,       0.04, 0.04],
    [0.12, 0.12,        0.105, 0.145, 0.135,       0.12, 0.12],
    [0.194, 0.194,        0.27, 0.194, 0.194,       0.194, 0.194],
    [0.228, 0.228,        0.16, 0.225, 0.228,       0.228, 0.228],
    [0.250, 0.250,        0.135, 0.245, 0.250,       0.250, 0.250],
]...)'

cps_halfThicknesses = hcat([
    [0.097, 0.097,        0.1005, 0.0875, 0.095,       0.097, 0.097],
    [0.097, 0.097,        0.1005, 0.0875, 0.095,       0.097, 0.097],
    [0.083, 0.083,        0.11, 0.0785, 0.08,       0.083, 0.083],
    [0.063, 0.063,        0.075, 0.045, 0.055,       0.063, 0.063],
    [0.0335, 0.0335,        0.045, 0.0345, 0.034,       0.0335, 0.0335],
    [0.011, 0.011,        0.011, 0.011, 0.011,       0.011, 0.011],
]...)'

yArr = thetas
cps_arr = cps_thetas

# yArr = lengths
# cps_arr = cps_lengths

# yArr = halfThicknesses
# cps_arr = cps_halfThicknesses

color_range = [RGB(get(ColorSchemes.algae, k)) for k in range(0, 1, length=length(tSplineCps))]

p = plot(dpi=200)
for i in 1:size(cps_arr, 1)
    # Data to be regresed.
    t = hcat([timeVals..., 1.])
    y = hcat([yArr[i, :]..., yArr[i, 1]])
    
    # Pick control points from the array and make a spline
    cps_y = cps_arr[i, :]
    cps = hcat([tSplineCps, cps_y]...)'
    pu = pbSpline(cps, s)

    # Plot
    plot!(pu[1, :], pu[2, :], label="", linewidth=3, color=color_range[i])
    plot!(cps[1, :], cps[2, :], linestyle=:dash, marker=:circle, label="", color=color_range[i])
    plot!(t, y, linestyle=:dash, marker=:square, label="", color=color_range[i])
end
plot!(show=true)




#=
# Add the first element as last to complete the cycle. Repeat at small offsets
# to force zero gradient to the curves and make for a smooth transition.
halfThicknesses = hcat(halfThicknesses[:, 1:1], halfThicknesses, halfThicknesses[:, 1:1], halfThicknesses[:, 1:1])
lengths = hcat(lengths[:, 1:1], lengths, lengths[:, 1:1], lengths[:, 1:1])
thetas  = hcat(thetas[:, 1:1], thetas, thetas[:, 1:1], thetas[:, 1:1])

# Create a layout with 3 rows and 1 column
layout = @layout([a; b; c])
p = plot(layout=(3, 1), dpi=200, size=(1200, 800), show=true)
color_range = [RGB(get(ColorSchemes.algae, k)) for k in range(0, 1, length=size(lengths, 2))]
for i in 1:size(lengths, 1)
    plot!(timeVals, lengths[i, :], linestyle=:dash, marker=:circle,
        markersize=2, alpha=0.2, label="", ylabel="Length", subplot=1, color=color_range[i])
    plot!(timeVals, halfThicknesses[i, :], linestyle=:dash, marker=:circle,
        markersize=2, alpha=0.2, label="", ylabel="Halfthickness", subplot=2, color=color_range[i])
    plot!(timeVals, thetas[i, :], linestyle=:dash, marker=:circle,
        markersize=2, alpha=0.2, label="", ylabel="Angle", subplot=3, color=color_range[i])
    
    # Smooth curves
    itp_l = interpolate(timeVals, lengths[i, :], SteffenMonotonicInterpolation())
    itp_t = interpolate(timeVals, halfThicknesses[i, :], SteffenMonotonicInterpolation())
    itp_a = interpolate(timeVals, thetas[i, :], SteffenMonotonicInterpolation())

    # Generate points for the smooth curves
    smooth_time = 0:0.01:1
    plot!(smooth_time, itp_l.(smooth_time), subplot=1, label="", color=color_range[i]) 
    plot!(smooth_time, itp_t.(smooth_time), subplot=2, label="", color=color_range[i]) 
    plot!(smooth_time, itp_a.(smooth_time), subplot=3, label="", color=color_range[i]) 
end

savefig("outputs/plot_01_bellShape_segmentParams.png")
=#

#=
# ===
# Bell shape.

Lfit, thkFit, thetaFit = params_for_profile(0.35, timeVals, lengths, halfThicknesses, thetas, aTarget=-1.0)
Lfit_o, thkFit_o, thetaFit_o = params_for_profile(0.35, timeVals, lengths, halfThicknesses, thetas)

xy, cps, area = profileFromParams(Lfit, thkFit, thetaFit)
xy_o, cps_o, area_o = profileFromParams(Lfit_o, thkFit_o, thetaFit_o)

plot(xy[1, :], xy[2, :], dpi=200, size=(1200, 800), label="Interpolated profile")
plot!(xy_o[1, :], xy_o[2, :], linestyle=:dashdot, label="Interpolated profile with area conservation enforced")
plot!(cps[1, :], cps[2, :], marker=:circle, linestyle=:dash, label="")
annotate!(0.05, -0.52, "A_target=\$$(round(0.0644, digits=4))\$ units\$^2\$", halign=:left)
annotate!(0.05, -0.55, "A=\$$(round(area, digits=4))\$ units\$^2\$", halign=:left)
annotate!(0.05, -0.58, "A_opt=\$$(round(area_o, digits=4))\$ units\$^2\$", halign=:left)
savefig("outputs/plot_02_bellShape_finalShape.png")

# ===
# Animate the shape for one cycle.

anim = @animate for i ∈ 1:101
    t = (i/101) % 1.0
    Lfit, thkFit, thetaFit = params_for_profile(t, timeVals, lengths, halfThicknesses, thetas)
    xy, cps, area = profileFromParams(Lfit, thkFit, thetaFit; mirror=true)
    plot(xy[1, :], xy[2, :], label="", color=:red, xlimits=(-0.75, 0.75), lw=2, ylimits=(-0.75, 0.01),
        aspect_ratio=:equal, xlabel="Bell width", ylabel="Bell height", dpi=200, size=(1200, 600))
    plot!(cps[1, :], cps[2, :], marker=:circle, linestyle=:dash, label="", color=:black, alpha=0.5, lw=2)
end
gif(anim, "outputs/plot_03_animatedBellShape.gif", fps=15)
=#

# TODO compare with parametric bodies
#=

    # define a flat plat at and angle of attack
    cps = SA[-1   0   1
            0.5 0.25 0]*L .+ [2L,3L]

    # needed if control points are moved
    cps_m = MMatrix(cps)
    # weights = SA[1.,1.,1.]
    # knots =   SA[0,0,0,1,1,1.]

    # make a nurbs curve
    # circle = NurbsCurve(cps_m,knots,weights)
    circle = BSplineCurve(cps_m;degree=2)
=#



