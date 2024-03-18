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
# Digitised from Costello plot.

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

#timeVals = [0, 0.01, 0.2, 0.4, 0.95, 1.0]
# timeVals = [0.04836558, 0.22848566, 0.41027352, 1.39926618, 1.57771848, 1.76617745];
timeVals = [0, 0.13, 0.27]

# Spline control points for regressing the motion of each jellyfish segment.
# We need an implementation compatible with GPUs. This necessitates the
# use of static arrays.

Nsegments = 6

cps_thetas = SMatrix{Nsegments+1, 7}(vcat([
# t/T values
[0, 0.025, 0.15, 0.25, 0.35, 0.45, 1.0]',
# CP locations
hcat([
    [0., 0.,        0., 0., 0.,       0., 0.],
    [0., 0.,        0., 0., 0.,       0., 0.],
    [0.39, 0.39,        0., 0.38, 0.39,       0.39, 0.39],
    [0.4, 0.4,        0.85, 1.2, 0.41,       0.4, 0.4],
    [0.95, 0.95,        1.95, 1.65, 0.95,       0.95, 0.95],
    [2.2, 2.2,        -1., 2.19, 2.2,       2.2, 2.2],
]...)'
]...));

cps_lengths = SMatrix{Nsegments+1, 7}(vcat([
[0, 0.025, 0.15, 0.25, 0.35, 0.45, 1.0]',
hcat([
    [0.02, 0.02,        0.01, 0.03, 0.022,       0.02, 0.02],
    [0.04, 0.04,        0.02, 0.06, 0.04,       0.04, 0.04],
    [0.12, 0.12,        0.1, 0.155, 0.11,       0.12, 0.12],
    [0.194, 0.194,        0.275, 0.194, 0.194,       0.194, 0.194],
    [0.228, 0.228,        0.16, 0.225, 0.228,       0.228, 0.228],
    [0.250, 0.250,        0.135, 0.245, 0.250,       0.250, 0.250],
]...)'
]...));

cps_halfThicknesses = SMatrix{Nsegments+1, 7}(vcat([
[0, 0.025, 0.15, 0.25, 0.35, 0.45, 1.0]',
hcat([
    [0.097, 0.097,        0.1005, 0.0875, 0.097,       0.097, 0.097],
    [0.097, 0.097,        0.1005, 0.0875, 0.097,       0.097, 0.097],
    [0.083, 0.083,        0.11, 0.075, 0.083,       0.083, 0.083],
    [0.063, 0.063,        0.075, 0.04, 0.063,       0.063, 0.063],
    [0.0335, 0.0335,        0.045, 0.0345, 0.034,       0.0335, 0.0335],
    [0.011, 0.011,        0.011, 0.011, 0.011,       0.011, 0.011],
]...)'
]...));

function smoothShapeParams(i, cps_arr, s)
    # Pick control points from the array and make a spline
    cps_y = cps_arr[[1, i], :]
    pu = old_evaluate_spline(cps_y, s)
    return cps_y, pu
end

color_range = [RGB(get(ColorSchemes.algae, k)) for k in range(0, 1, length=length(tSplineCps))];

p = plot(dpi=200, xlabel="t/T", ylabel="Segment thickness")
# Ignore the time values in the loop.
for i in 2:size(cps_thetas, 1)
    cps, pu = smoothShapeParams(i, cps_halfThicknesses, 0:0.01:1.0)
    plot!(pu[1, :], pu[2, :], label="", linewidth=3, color=color_range[i])
    plot!(cps[1, :], cps[2, :], linestyle=:dashdot, marker=:square, label="", color=color_range[i])
    plot!(vcat(timeVals, 1.0), vcat(halfThicknesses[i-1, :], halfThicknesses[i-1, 1]),
        linestyle=:dash, marker=:circle, label="", color=:red, markersize=5)
end

for i in 1:6
    filename = @sprintf("D:/git/JellyfishGiggles/dataset_01_medusae/smoothedShapeParams_segment_%d_Costello2020.txt", i-1)
    refData = readdlm(filename)
    plot!(refData[:, 1], refData[:, 3], linewidth=2, color=:black, linestyle=:dash, label="")
end

plot!(show=true)

p = plot(dpi=200, xlabel="t/T", ylabel="Segment length")

for i in 2:size(cps_thetas, 1)
    cps, pu = smoothShapeParams(i, cps_lengths, 0:0.01:1.0)
    plot!(pu[1, :], pu[2, :], label="", linewidth=3, color=color_range[i])
    plot!(cps[1, :], cps[2, :], linestyle=:dashdot, marker=:square, label="", color=color_range[i])
    plot!(vcat(timeVals, 1.0), vcat(lengths[i-1, :], lengths[i-1, 1]),
        linestyle=:dash, marker=:circle, label="", color=:red, markersize=5)
end

for i in 1:6
    filename = @sprintf("D:/git/JellyfishGiggles/dataset_01_medusae/smoothedShapeParams_segment_%d_Costello2020.txt", i-1)
    refData = readdlm(filename)
    plot!(refData[:, 1], refData[:, 2], linewidth=2, color=:black, linestyle=:dash, label="")
end

plot!(show=true)

p = plot(dpi=200, xlabel="t/T", ylabel="Segment angle")

for i in 2:size(cps_thetas, 1)
    cps, pu = smoothShapeParams(i, cps_thetas, 0:0.01:1.0)
    plot!(pu[1, :], pu[2, :], label="", linewidth=3, color=color_range[i])
    plot!(cps[1, :], cps[2, :], linestyle=:dashdot, marker=:square, label="", color=color_range[i])
    plot!(vcat(timeVals, 1.0), vcat(thetas[i-1, :], thetas[i-1, 1]),
        linestyle=:dash, marker=:circle, label="", color=:red, markersize=5)
end

for i in 1:6
    filename = @sprintf("D:/git/JellyfishGiggles/dataset_01_medusae/smoothedShapeParams_segment_%d_Costello2020.txt", i-1)
    refData = readdlm(filename)
    plot!(refData[:, 1], refData[:, 4], linewidth=2, color=:black, linestyle=:dash, label="")
end

plot!(show=true)

#

iSeg = 2
getSegmentPosition(iSeg, 0.21, cps_halfThicknesses, printout=true)

#

function shapeCostello(iCostello_a)
    x_ref_a = vcat([shapeData_Costello[iCostello_a][:, 1], shapeData_Costello[iCostello_a][:, 3]]...)
    y_ref_a = vcat([shapeData_Costello[iCostello_a][:, 2], shapeData_Costello[iCostello_a][:, 4]]...)
    x0_a = (findmax(x_ref_a)[1] + findmin(x_ref_a)[1]) / 2.0
    y0_a = findmax(y_ref_a)[1]
    return x_ref_a .- x0_a, y_ref_a .- y0_a
end

plot(xlabel="x/L", ylabel="y/L", aspect_ratio=:equal, size=(1000, 450))

# ===
xy, cps, segParams1 = shapeForTime(0.)
x_ref_a, y_ref_a = shapeCostello(1)
x_ref_b, y_ref_b = shapeCostello(4)

plot!(abs.(x_ref_a), y_ref_a, marker=:dot, linewidth=0, color=:red, markersize=2,
    label="Costello et al., picture 1, t/T=0.00")
plot!(abs.(x_ref_b), y_ref_b, marker=:v, linewidth=0, color=:red, markersize=2,
    label="Costello et al., picture 1, t/T=0.00")
plot!(xy[1, :], xy[2, :], linewidth=2, color=:red, label="Current model, t/T=0.00")
plot!(cps[1, :], cps[2, :], linewidth=2, color=:red, marker=:square, linestyle=:dash, markersize=4, label="")

shape_Costello_regressed = readdlm("D:/git/JellyfishGiggles/dataset_01_medusae/shape_Costello2020_snapshot1.txt")
refCps = readdlm("D:/git/JellyfishGiggles/dataset_01_medusae/smoothShapeCps_Costello2020_snapshot0.txt")
plot!(shape_Costello_regressed[:, 1], shape_Costello_regressed[:, 2], linewidth=2, color=:black, label="")
plot!(refCps[:, 1] .+ 0.0, refCps[:, 2], linewidth=2, marker=:x, linestyle=:dashdotdot, color=:black,
    markersize=5, label="")

# ===
xy, cps, segParams2 = shapeForTime(0.13)
x_ref_a, y_ref_a = shapeCostello(2)
x_ref_b, y_ref_b = shapeCostello(5)

plot!(abs.(x_ref_a) .+ 0.5, y_ref_a, marker=:dot, linewidth=0, color=:green, markersize=2,
    label="Costello et al., picture 1, t/T=0.13")
plot!(abs.(x_ref_b) .+ 0.5, y_ref_b, marker=:v, linewidth=0, color=:green, markersize=2,
    label="Costello et al., picture 1, t/T=0.13")
plot!(xy[1, :] .+ 0.5, xy[2, :], linewidth=2, color=:green, label="Current model, t/T=0.13")
plot!(cps[1, :] .+ 0.5, cps[2, :], linewidth=2, color=:green, marker=:square, linestyle=:dash, markersize=4, label="")

shape_Costello_regressed = readdlm("D:/git/JellyfishGiggles/dataset_01_medusae/shape_Costello2020_snapshot2.txt")
refCps = readdlm("D:/git/JellyfishGiggles/dataset_01_medusae/smoothShapeCps_Costello2020_snapshot1.txt")
plot!(shape_Costello_regressed[:, 1], shape_Costello_regressed[:, 2], linewidth=2, color=:black, label="")
plot!(refCps[:, 1] .+ 0.5, refCps[:, 2], linewidth=2, marker=:x, linestyle=:dashdotdot, color=:black,
    markersize=5, label="")

# ===
xy, cps, segParams3 = shapeForTime(0.27)
x_ref_a, y_ref_a = shapeCostello(4)
x_ref_b, y_ref_b = shapeCostello(6)

plot!(abs.(x_ref_a) .+ 1.0, y_ref_a, marker=:dot, linewidth=0, color=:blue, markersize=2,
    label="Costello et al., picture 1, t/T=0.27")
plot!(abs.(x_ref_b) .+ 1.0, y_ref_b, marker=:v, linewidth=0, color=:blue, markersize=2,
    label="Costello et al., picture 1, t/T=0.27")
plot!(xy[1, :] .+ 1.0, xy[2, :], linewidth=2, color=:blue, label="Current model, t/T=0.27")
plot!(cps[1, :] .+ 1.0, cps[2, :], linewidth=2, color=:blue, marker=:square, linestyle=:dash, markersize=4, label="")

shape_Costello_regressed = readdlm("D:/git/JellyfishGiggles/dataset_01_medusae/shape_Costello2020_snapshot3.txt")
refCps = readdlm("D:/git/JellyfishGiggles/dataset_01_medusae/smoothShapeCps_Costello2020_snapshot2.txt")
plot!(shape_Costello_regressed[:, 1], shape_Costello_regressed[:, 2], linewidth=2, color=:black, label="")
plot!(refCps[:, 1] .+ 1.0, refCps[:, 2], linewidth=2, marker=:x, linestyle=:dashdotdot, color=:black,
    markersize=5, label="")

plot!(legend=:outertop, legend_columns=3, show=true)

#

# Function to generate frames for the animation
function generate_frames()
    anim = Animation()
    for x in 0:0.01:1
        xy, cps, segParams = shapeForTime(x)
        fs = 14
        plot(xlabel="x/L", ylabel="y/L", aspect_ratio=:equal, xlims=(0, 0.75), ylims=(-0.75, 0), size=(1000, 800),
            xtickfontsize=fs, ytickfontsize=fs, xguidefontsize=fs, yguidefontsize=fs, legendfontsize=fs)

        shape_Costello_regressed = readdlm("D:/git/JellyfishGiggles/dataset_01_medusae/shape_Costello2020_snapshot1.txt")
        plot!(shape_Costello_regressed[:, 1], shape_Costello_regressed[:, 2], linewidth=2, color=:red,
            linestyle=:dash, label="Costello et al., t/T=0.00")

        shape_Costello_regressed = readdlm("D:/git/JellyfishGiggles/dataset_01_medusae/shape_Costello2020_snapshot2.txt")
        plot!(shape_Costello_regressed[:, 1].-0.5, shape_Costello_regressed[:, 2], linewidth=2, color=:green,
            linestyle=:dash, label="Costello et al., t/T=0.13")

        shape_Costello_regressed = readdlm("D:/git/JellyfishGiggles/dataset_01_medusae/shape_Costello2020_snapshot3.txt")
        plot!(shape_Costello_regressed[:, 1].-1.0, shape_Costello_regressed[:, 2], linewidth=2, color=:blue,
            linestyle=:dash, label="Costello et al., t/T=0.27")

        plot!(xy[1, :], xy[2, :], color=:black, linewidth=3, label="")
        plot!(cps[1, :], cps[2, :], linewidth=2, color=:black, marker=:circle, linestyle=:dash, markersize=4, alpha=0.5, label="")

        formatted_float = @sprintf("%.2f", x)
        annotate!(0.65, -0.7, text("t/T=$formatted_float", fs, :black))

        frame(anim)
    end
    return anim
end

# Generate frames
anim = generate_frames()

# Save the animation as a GIF
gif(anim, "jelly.gif", fps=10)

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
