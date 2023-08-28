using Plots
using Images
using LinearAlgebra
using StaticArrays
using Interpolations
using ColorSchemes
using Optim

function coxDeBoor(knots, u, k, d, count)
    """
        coxDeBoor(knots, u, k, d, count)

    Compute the Cox-De Boor recursion for B-spline basis functions.

    The `coxDeBoor` function computes the Cox-De Boor recursion for B-spline basis functions,
    used in the evaluation of B-spline curves and surfaces.

    Arguments:
    - `knots`: An array of knot values.
    - `u`: The parameter value at which to evaluate the B-spline basis function.
    - `k`: The index of the current knot interval.
    - `d`: The degree of the B-spline basis function.
    - `count`: The number of control points.

    Returns:
    The value of the B-spline basis function at parameter `u` and knot interval `k`.
    """
    if (d == 0)
        return Int(((knots[k+1] <= u) && (u < knots[k+2])) || ((u >= (1.0-1e-12)) && (k == (count-1))))
    end
    return (((u-knots[k+1])/max(1e-12, knots[k+d+1]-knots[k+1]))*coxDeBoor(knots, u, k, (d-1), count)
        + ((knots[k+d+2]-u)/max(1e-12, knots[k+d+2]-knots[k+2]))*coxDeBoor(knots, u, (k+1), (d-1), count))
end

function bspline(cv, s; d=3)
    """
        bspline(cv, s; d=3)

    Evaluate a B-spline curve at a given parameter value.

    The `bspline` function evaluates a B-spline curve at the specified parameter `s`.

    Arguments:
    - `cv`: A 2D array representing the control points of the B-spline curve.
    - `s`: The parameter value at which the B-spline curve should be evaluated.
    - `d`: The degree of the B-spline curve (default is 3).

    Returns:
    A vector representing the point on the B-spline curve at parameter `s`.

    Note:
    - This function assumes a column-major orientation of points as Julia gods intended.
    """
    count = size(cv, 2)
    knots = vcat(zeros(d), range(0, count-d) / (count-d), ones(d))
    pt = zeros(size(cv, 1))
    for k in range(0, count-1)
        pt += coxDeBoor(knots, s, k, d, count) * cv[:, k+1]
    end
    return pt
end

function evaluate_spline(cps, s)
    """
        evaluate_spline(cps, s)

    Evaluate a B-spline curve at multiple parameter values.

    The `evaluate_spline` function evaluates a B-spline curve at the specified parameter values `s`.

    Arguments:
    - `cps`: A 2D array representing the control points of the B-spline curve.
    - `s`: An array of parameter values at which the B-spline curve should be evaluated.

    Returns:
    A 2D array where each column corresponds to a point on the B-spline curve at the parameter values in `s`.

    Note:
    - This function assumes a column-major orientation of points as Julia gods intended.
    """
    return hcat([bspline(cps, u, d=2) for u in s]...)
end

function polyArea(xy)
    """
        polyArea(xy)

    Calculate the area of a polygon using the Shoelace formula.

    The `polyArea` function calculates the area of a polygon defined by its vertices using the Shoelace formula.

    Arguments:
    - `xy`: A 2xN matrix containing the (x, y) coordinates of the polygon's vertices. They do not need to
        form a closed shape, but it is assumed that the first and last points are connected by a straight line.

    Returns:
    The area of the polygon.
    """
    return 0.5*abs(dot(xy[1, :], [xy[2, 2:end]..., xy[2, 1]]) - dot(xy[2, :], [xy[1, 2:end]..., xy[1, 1]]))
end

function rotatePoint(pt, x0, theta)
    """
        rotatePoint(pt, x0, theta)

    Rotate a point around a specified point by a given angle.

    The `rotatePoint` function rotates a point `pt` around the point `x0` by the specified angle `theta`.

    Arguments:
    - `pt`: A 2-element vector representing the point to be rotated.
    - `x0`: A 2-element vector representing the center of rotation.
    - `theta`: The angle of rotation in radians.

    Returns:
    A 2-element vector representing the rotated point.
    """
    x = pt - x0
    xr = x[1]*cos(theta) - x[2]*sin(theta)
    yr = x[2]*cos(theta) + x[1]*sin(theta)
    return [xr, yr] + x0
end

function profileFromParams(lengths, halfThickness, theta, s=0:0.01:1)
    """
        profileFromParams(lengths, halfThickness, theta, s=0:0.01:1)

    Calculate profile coordinates, control points, and area based on input parameters.

    Arguments
    - `lengths::Vector{Float64}`: Array of lengths.
    - `halfThickness::Vector{Float64}`: Array of half thicknesses.
    - `theta::Vector{Float64}`: Array of angles in radians.
    - `s::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}`: Parameter range for evaluation. Default is `0:0.01:1`.

    Returns
    - `Array{Float64, 2}`: Array of profile coordinates.
    - `Array{Float64, 2}`: Array of control points.
    - `Float64`: Calculated area.

    """
    cps_u = zeros(2, size(lengths, 1))
    cps_l = zeros(2, size(lengths, 1))

    xLast = [-lengths[1]/2, -halfThickness[1]]
    
    for i in 1:length(halfThickness)
        xNew = rotatePoint(xLast + [lengths[i], 0], xLast, -theta[i])
        xMid = (xLast + xNew) / 2.0
        vTan = (xNew - xLast) / norm(xNew - xLast)
        vPer = [-vTan[2], vTan[1]]

        cps_u[:, i] = xMid + halfThickness[i]*vPer
        cps_l[:, i] = xMid - halfThickness[i]*vPer
        
        xLast = xNew
    end

    cps = hcat(cps_u, reverse(cps_l, dims=2))
    xy = evaluate_spline(cps, s)
    area = polyArea(xy)
    
    return xy, cps, area
end

function thicknessTarget(x, Lfit, thkFit, thetaFit, aTarget=0.0644)
    """
        thicknessTarget(x, Lfit, thkFit, thetaFit)

    Calculate the absolute difference between the calculated area and a target area
    for a thickness increment x in range 0 and 1.

    Arguments
    - `x::Float64`: thickness increment.
    - `Lfit::Vector{Float64}`: List of segment lengths.
    - `thkFit::Vector{Float64}`: List of segment thicknesses
    - `thetaFit::Vector{Float64}`: List of segment angles
    - `aTarget::Float64`: target area, default is 0.0644 (from Costello 2020).

    Returns
    - `Float64`: Absolute difference between the calculated area and the target area.

    """

    dt = (x - 0.5) / 0.5 * 0.01
    xy1, cps1, area1 = profileFromParams(Lfit, thkFit .+ dt, thetaFit)
    return abs(area1 - aTarget)
end

# ===
# Test splines
s = 0:0.01:1

# 1st cycle, upper
cps = hcat(
    [0, 0.],
    [0.05, 0],
    [0.472, -0.25],
    [0.389, -0.52]
)

pu = evaluate_spline(cps, s)
plot(pu[1, :], pu[2, :], label="", color=:blue)
plot!(cps[1, :], cps[2, :], linestyle=:dash, marker=:circle, label="", color=:blue)

# 1st cycle lower
cps = hcat(
    [0, -0.2],
    [0.15, -0.2],
    [0.4, -0.35],
    [0.389, -0.52]
)

pl = evaluate_spline(cps, s)
plot!(pl[1, :], pl[2, :], label="", color=:red)
plot!(cps[1, :], cps[2, :], linestyle=:dash, marker=:circle, label="", color=:red)

xy = hcat(pu, reverse(pl[:, 1:end-1], dims=2))
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

xy, cps, area = profileFromParams(lengths[:, 1], halfThicknesses[:, 1], thetas[:, 1])

plot(xy[1, :], xy[2, :], dpi=200, size=(1200, 800), labels="")
plot!(cps[1, :], cps[2, :], marker=:circle, linestyle=:dash, label="")
annotate!(0.1, 0., "A=\$$(round(area, digits=4))\$ units\$^2\$", valign=:bottom, halign=:left)
savefig("outputs/plot_02_bellShape_finalShape.png")

# ===
# Thickness optimisation for volume conservation


# TODO need to interpolate for a given t
f(x) = thicknessTarget(x[1], lengths[:, 1], halfThicknesses[:, 1], thetas[:, 1])#Lfit, thkFit, thetaFit)

result = optimize(f, [0.0], [1.0], [0.5])

# Get the optimized x value
x_optimized = result.minimizer[1]

println("Optimized x value: $x_optimized")


#=
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
=#
