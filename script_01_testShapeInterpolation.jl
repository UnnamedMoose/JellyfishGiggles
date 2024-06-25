using Printf
using Plots
using Images
using LinearAlgebra
using StaticArrays
using Interpolations
using ColorSchemes
using Optim
using DelimitedFiles

using ParametricBodies
# include("D:/git/ParametricBodies.jl/src/NurbsCurves.jl")

#include("./src/JellyfishPhysics.jl")
#using .JellyfishPhysics

include("./src/splines.jl")
include("./src/kinematics.jl")

# ===
s = 0:0.01:1

plotPython = false

# ===
# Smooth interpolation test
kinematicsArr = kinematics_2_largeFlick
t = 0.13

seg_theta = []
seg_length = []
seg_thickness = []
for iSeg in 1:size(kinematicsArr.cps_thetas, 1)-1
    push!(seg_theta, getSegmentPosition(iSeg, t, kinematicsArr.cps_thetas))
#    push!(seg_length, getSegmentPosition(iSeg, t, kinematicsArr.cps_lengths))
#    push!(seg_thickness, getSegmentPosition(iSeg, t, kinematicsArr.cps_halfThicknesses))
end

println("")
println(seg_theta)
println("")
for iSeg in 1:size(kinematicsArr.cps_thetas, 1)
    println(kinematicsArr.cps_thetas[iSeg, :])
end
println("")

sx = 0.5
iSeg = 6
# Pick control points from the array and make a spline
cps_y = kinematicsArr.cps_thetas[[1, iSeg+1], :]
#t, y = old_evaluate_spline(cps_y, [sx])
t, y = pbSpline(cps_y, [sx])
println(cps_y[1, :])
println(cps_y[2, :])
println(t, " ", y)

s = 0:0.01:1
y = pbSpline(cps_y, s)
y_test = readdlm("./testData/segSpline.txt")'
plot(cps_y[1, :], cps_y[2, :], color=:black, marker=:circle, linestyle=:dashdot, linewidth=1, label="CPs")
plot!(y_test[1, :], y_test[2, :], color=:red, linewidth=3, label="Python")
plot!(y[1, :], y[2, :], show=false, color=:blue, linewidth=3, linestyle=:dash, label="Julia")
savefig("outputs/plot_01_bellShape_segmentParams_splinesComparison.png")

# ===
# Additional kinematics checks.
xy_test = readdlm("./testData/xy.txt")'
cps_test = readdlm("./testData/cps.txt")'
backbone_test = readdlm("./testData/backbone.txt")'
xy, cps_x, segParams, backbone = shapeForTime(0.13, kinematics_2_largeFlick)

plot(xlabel="x/L", ylabel="y/L", aspect_ratio=:equal, xlims=(0, 0.75), ylims=(-0.75, 0), size=(1000, 800))

plot!(xy[1, :], xy[2, :], color=:red, linewidth=3, label="Shape - Julia")
plot!(xy_test[1, :], xy_test[2, :], color=:red, linestyle=:dashdot, linewidth=3, label="Shape - Python")

plot!(cps_x[1, :], cps_x[2, :], color=:black, marker=:circle, linewidth=1, label="cps - Julia")
plot!(cps_test[1, :], cps_test[2, :], color=:black, marker=:circle, linestyle=:dashdot, linewidth=1, label="cps - Python")

plot!(backbone[1, :], backbone[2, :], color=:blue, marker=:square, linewidth=3, label="Backbone - Julia")
plot!(backbone_test[1, :], backbone_test[2, :], color=:blue, marker=:square, linestyle=:dashdot, linewidth=3, label="Backbone - Python")

savefig("outputs/plot_01_bellShape_staticShape_T_0.13.png")
plot!(show=false)

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

# Function for smoothing the shape parameters. Used for plotting only.
function smoothShapeParams(iSeg, cps_arr, s)
    # Pick control points from the array and make a spline
    cps_y = cps_arr[[1, iSeg+1], :]
    # pu = old_evaluate_spline(cps_y, s)
    pu = pbSpline(cps_y, s)
    return cps_y, pu
end

color_range = [RGB(get(ColorSchemes.algae, k)) for k in range(0, 1, length=kinematics_0_baseline.Nsegments)];

# ===
p = plot(dpi=200, xlabel="t/T", ylabel="Segment thickness")
# Ignore the time values in the loop.
for i in 1:kinematics_0_baseline.Nsegments
    cps, pu = smoothShapeParams(i, kinematics_0_baseline.cps_halfThicknesses, 0:0.01:1.0)
    if i == 1
        label1 = "Shape"; label2 = "CPs"; label3 = "Costello"
    else
        label1 = ""; label2 = ""; label3 = ""
    end
    plot!(pu[1, :], pu[2, :], label=label1, linewidth=3, color=color_range[i])
    plot!(cps[1, :], cps[2, :], linestyle=:dashdot, marker=:square, label=label2, color=color_range[i])
    plot!(vcat(timeVals, 1.0), vcat(halfThicknesses[i, :], halfThicknesses[i, 1]),
        linestyle=:dash, marker=:circle, label=label3, color=:red, markersize=5)
end

if plotPython
    for i in 1:6
        filename = @sprintf("./dataset_01_medusae/smoothedShapeParams_segment_%d_Costello2020.txt", i-1)
        refData = readdlm(filename)
        if i == 1
            label = "Python"
        else
            label = ""
        end
        plot!(refData[:, 1], refData[:, 3], linewidth=2, color=:black, linestyle=:dash, label=label)
    end
end

plot!(legend=:outertop, legend_columns=3, show=false)
savefig("outputs/plot_01_bellShape_segmentParams_thickness.png")

# ===
p = plot(dpi=200, xlabel="t/T", ylabel="Segment length")

for i in 1:kinematics_0_baseline.Nsegments
    cps, pu = smoothShapeParams(i, kinematics_0_baseline.cps_lengths, 0:0.01:1.0)
    if i == 1
        label1 = "Shape"; label2 = "CPs"; label3 = "Costello"
    else
        label1 = ""; label2 = ""; label3 = ""
    end
    plot!(pu[1, :], pu[2, :], label=label1, linewidth=3, color=color_range[i])
    plot!(cps[1, :], cps[2, :], linestyle=:dashdot, marker=:square, label=label2, color=color_range[i])
    plot!(vcat(timeVals, 1.0), vcat(lengths[i, :], lengths[i, 1]),
        linestyle=:dash, marker=:circle, label=label3, color=:red, markersize=5)
end

if plotPython
    for i in 1:6
        filename = @sprintf("./dataset_01_medusae/smoothedShapeParams_segment_%d_Costello2020.txt", i-1)
        refData = readdlm(filename)
        if i == 1
            label = "Python"
        else
            label = ""
        end
        plot!(refData[:, 1], refData[:, 2], linewidth=2, color=:black, linestyle=:dash, label=label)
    end
end

plot!(legend=:outertop, legend_columns=3, show=false)
savefig("outputs/plot_01_bellShape_segmentParams_length.png")

# ===
p = plot(dpi=200, xlabel="t/T", ylabel="Segment angle")

for i in 1:kinematics_0_baseline.Nsegments
    cps, pu = smoothShapeParams(i, kinematics_0_baseline.cps_thetas, 0:0.01:1.0)
    if i == 1
        label1 = "Shape"; label2 = "CPs"; label3 = "Costello"
    else
        label1 = ""; label2 = ""; label3 = ""
    end
    plot!(pu[1, :], pu[2, :], label=label1, linewidth=3, color=color_range[i])
    plot!(cps[1, :], cps[2, :], linestyle=:dashdot, marker=:square, label=label2, color=color_range[i])
    plot!(vcat(timeVals, 1.0), vcat(thetas[i, :], thetas[i, 1]),
        linestyle=:dash, marker=:circle, label=label3, color=:red, markersize=5)
end

if plotPython
    for i in 1:6
        filename = @sprintf("./dataset_01_medusae/smoothedShapeParams_segment_%d_Costello2020.txt", i-1)
        refData = readdlm(filename)
        if i == 1
            label = "Python"
        else
            label = ""
        end
        plot!(refData[:, 1], refData[:, 4], linewidth=2, color=:black, linestyle=:dash, label=label)
    end
end

plot!(legend=:outertop, legend_columns=3, show=false)
savefig("outputs/plot_01_bellShape_segmentParams_angle.png")


