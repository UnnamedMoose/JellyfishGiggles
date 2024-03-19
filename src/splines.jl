using Plots
using Images
using LinearAlgebra
using StaticArrays
using Interpolations
using ColorSchemes
using Optim
using StaticArrays
using CUDA

using ParametricBodies

function pbSpline(cps, s; deg=2)
    """ Handy wrapper around ParametricBodies.jl spline class. """
    cps_m = MMatrix(SArray{Tuple{2, size(cps[1,:])[1]}}(cps))
    curve = BSplineCurve(cps_m; degree=deg)
    return hcat([curve(u, 0) for u in s]...)
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

function getSegmentPosition(iSeg, tTarget, cps; NiterMax=100, tol=1e-6, printout=false)
    # For an arbitrary t/T value, need to either pre-compute the splines describing
    # the motion of each segment at the right time values or have an iterative routine
    # for matching the desired time.
    s0 = 0.
    s1 = 1.
    y = 0.

    for i in 1:1:NiterMax
        # Evaluate at the centre of the trust region.
        sx = (s0 + s1) / 2.

        # Pick control points from the array and make a spline
        cps_y = cps[[1, iSeg+1], :]
        #t, y = old_evaluate_spline(cps_y, [sx])
        t, y = pbSpline(cps_y, [sx])

        if printout
            println(i, " ", abs(t-tTarget), " ", t)
        end

        # Check if converge. Shrink the trust region if not.
        if abs(t-tTarget) < tol
            return y
        end
        if t > tTarget
            s1 = sx
        else
            s0 = sx
        end
    end

    # Return the best available approximation even though convergence has not been met.
    return y
end

function shapeForTime(t; mirror=false, evaluate=true, s=0:0.01:1)
    # Get parameter values for this point in the cycle.
    seg_theta = []
    seg_length = []
    seg_thickness = []
    for iSeg in 1:size(cps_thetas, 1)-1
        push!(seg_theta, getSegmentPosition(iSeg, t, cps_thetas))
        push!(seg_length, getSegmentPosition(iSeg, t, cps_lengths))
        push!(seg_thickness, getSegmentPosition(iSeg, t, cps_halfThicknesses))
            # old_evaluate_spline(hcat([tSplineCps, cps_thetas[i, :]]...)', [t])[2])
    end

    # Construct the control points from the params.
    cps_u = zeros(2, size(seg_length, 1))
    cps_l = zeros(2, size(seg_length, 1))

    xLast = [-seg_length[1]/2, -seg_thickness[1]]

    for i in 1:length(seg_thickness)
        xNew = rotatePoint(xLast + [seg_length[i], 0], xLast, -seg_theta[i])
        xMid = (xLast + xNew) / 2.0
        vTan = (xNew - xLast) / norm(xNew - xLast)
        vPer = [-vTan[2], vTan[1]]

        cps_u[:, i] = xMid + seg_thickness[i]*vPer
        cps_l[:, i] = xMid - seg_thickness[i]*vPer

        xLast = xNew
    end

    cps = hcat(cps_u, reverse(cps_l, dims=2))
    if mirror
        cps = hcat(cps, reverse(cps[:, 1:end-1].*[-1.0, 1.0], dims=2))
    end
    
    if evaluate
        #xy = old_evaluate_spline(cps, s)
        xy = pbSpline(cps, s)
        return xy, cps, hcat([seg_length, seg_thickness, seg_theta]...)
    else
        return cps
    end
end


# TODO obsolete - now using ParametricBodies.jl
function old_coxDeBoor(knots, u, k, d, count)
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
    return (((u-knots[k+1])/max(1e-12, knots[k+d+1]-knots[k+1]))*old_coxDeBoor(knots, u, k, (d-1), count)
        + ((knots[k+d+2]-u)/max(1e-12, knots[k+d+2]-knots[k+2]))*old_coxDeBoor(knots, u, (k+1), (d-1), count))
end

# TODO obsolete - now using ParametricBodies.jl
function old_bspline(cv, s; d=3)
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
        pt += old_coxDeBoor(knots, s, k, d, count) * cv[:, k+1]
    end
    return pt
end

# TODO obsolete
function xold_evaluate_spline(cps, s)
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
    return hcat([old_bspline(cps, u, d=2) for u in s]...)
end

function profileFromParams(lengths, halfThickness, theta; s=0:0.01:1, mirror=false)
    """
        profileFromParams(lengths, halfThickness, theta; s=0:0.01:1)

    Calculate profile coordinates, control points, and area based on input parameters.

    Arguments
    - `lengths::Vector{Float64}`: Array of lengths.
    - `halfThickness::Vector{Float64}`: Array of half thicknesses.
    - `theta::Vector{Float64}`: Array of angles in radians.
    - `s::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}`: Parameter range for evaluation. Default is `0:0.01:1`.
    - `mirror::{Bool}`: whether or not to make a full profile, default is false.

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
    if mirror
        cps = hcat(cps, reverse(cps[:, 1:end-1].*[-1.0, 1.0], dims=2))
    end
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

function params_for_profile(tOverT, timeVals, lengths, halfThicknesses, thetas; aTarget=0.0644)
    """
        params_for_profile(tOverT, timeVals, lengths, halfThicknesses, thetas)

    Calculate parameters for profile control points at a non-dimensional time `tOverT`
    using smooth spline interpolation given source data at sparse time values.

    # Arguments
    - `tOverT::Float64`: Non-dimensional time value.
    - `timeVals::Vector{Float64}`: Array of time values.
    - `lengths::Matrix{Float64}`: Matrix of length values for each control point over time.
    - `halfThicknesses::Matrix{Float64}`: Matrix of half thickness values for each control point over time.
    - `thetas::Matrix{Float64}`: Matrix of angle values for each control point over time.
    - `aTarget::Float64`: target area for the profile, default is 0.0644 (from Costello 2020).
        Set to a negative value in order to disable optimisation. The area of the profile will then not
        be exactly the same for all t/T parameter values.

    # Returns
    - `Vector{Float64}`: Array of fitted length values for the profile at `tOverT`.
    - `Vector{Float64}`: Array of fitted half thickness values for the profile at `tOverT`.
    - `Vector{Float64}`: Array of fitted angle values for the profile at `tOverT`.

    """
    Lfit, thkFit, thetaFit = zeros(size(lengths, 1)), zeros(size(lengths, 1)), zeros(size(lengths, 1))
    for i in 1:length(Lfit)
        itp_l = interpolate(timeVals, lengths[i, :], SteffenMonotonicInterpolation())
        itp_t = interpolate(timeVals, halfThicknesses[i, :], SteffenMonotonicInterpolation())
        itp_a = interpolate(timeVals, thetas[i, :], SteffenMonotonicInterpolation())

        Lfit[i] = itp_l(tOverT)
        thkFit[i] = itp_t(tOverT)
        thetaFit[i] = itp_a(tOverT)
    end

    if aTarget > 0.0
        f(x) = thicknessTarget(x[1], Lfit, thkFit, thetaFit)
        result = optimize(f, [0.0], [1.0], [0.5])
        dThick = (result.minimizer[1] - 0.5)/0.5 * 0.01
        thkFit .+= dThick
    end

    return Lfit, thkFit, thetaFit
end
