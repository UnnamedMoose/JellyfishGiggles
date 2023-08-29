using Plots
using Images
using LinearAlgebra
using StaticArrays
using Interpolations
using ColorSchemes
using Optim

include("./src/JellyfishPhysics.jl")
using .JellyfishPhysics

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

pu = evaluate_spline(cps_u, s)

# 1st cycle lower
cps_l = hcat(
    [0, -0.2],
    [0.15, -0.2],
    [0.4, -0.35],
    [0.389, -0.52]
)

pl = evaluate_spline(cps_l, s)
xy = hcat(pu, reverse(pl[:, 1:end-1], dims=2))

plot(pu[1, :], pu[2, :], label="", color=:blue)
plot!(cps_u[1, :], cps_u[2, :], linestyle=:dash, marker=:circle, label="", color=:blue)
plot!(pl[1, :], pl[2, :], label="", color=:red)
plot!(cps_l[1, :], cps_l[2, :], linestyle=:dash, marker=:circle, label="", color=:red)
plot!(xy[1, :], xy[2, :], color=:black, lw=2, linestyle=:dash)


# ===
# Test area computation.
println(polyArea(xy))
println(0.06645296399999623)

# ===
# Test rotation
cps = hcat(
    rotatePoint([0, -0.2], [0., -0.2], 15.0/180.0*pi),
    rotatePoint([0.15, -0.2], [0., -0.2], 15.0/180.0*pi),
    rotatePoint([0.4, -0.35], [0., -0.2], 15.0/180.0*pi),
    rotatePoint([0.389, -0.52], [0., -0.2], 15.0/180.0*pi),
)
pl = evaluate_spline(cps, s)
plot!(pl[1, :], pl[2, :], label="", color=:orange)
plot!(cps[1, :], cps[2, :], linestyle=:dash, marker=:circle, label="", color=:orange)

savefig("outputs/plot_00_bellShape_testSplinesAndRotation.png")

# ===
# Bell shape parameters.
halfThicknesses = hcat(
    [0.101, 0.099, 0.090],
    [0.081, 0.102, 0.077],
    [0.063, 0.069, 0.048],
    [0.033, 0.041, 0.0338],
    [0.011, 0.011, 0.011],
)'

lengths = hcat(
    [0.080, 0.052, 0.103],
    [0.118, 0.105, 0.163],
    [0.194, 0.250, 0.194],
    [0.228, 0.179, 0.228],
    [0.250, 0.165, 0.250],
)'

thetas = hcat(
    [0., 0., 0.],
    [0.380, 0.080, 0.250],
    [0.400, 0.731, 1.100],
    [0.900, 1.670, 1.370],
    [2.200, 0.125, 2.100],
)'

# TODO digitise from Costello plot.
timeVals = [0, 0.01, 0.2, 0.4, 0.95, 1.0]

# Add the first element as last to complete the cycle. Repeat at small offsets
# to force zero gradient to the curves and make for a smooth transition.
halfThicknesses = hcat(halfThicknesses[:, 1:1], halfThicknesses, halfThicknesses[:, 1:1], halfThicknesses[:, 1:1])
lengths = hcat(lengths[:, 1:1], lengths, lengths[:, 1:1], lengths[:, 1:1])
thetas  = hcat(thetas[:, 1:1], thetas, thetas[:, 1:1], thetas[:, 1:1])

# Create a layout with 3 rows and 1 column
layout = @layout([a; b; c])
p = plot(layout=(3, 1), dpi=200, size=(1200, 800))
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

