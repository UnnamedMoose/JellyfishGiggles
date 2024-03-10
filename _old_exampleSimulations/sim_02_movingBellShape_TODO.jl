# Custom bell shape routines
include("./src/JellyfishPhysics.jl")
using .JellyfishPhysics

# WaterLily and misc addons
using WaterLily
using ParametricBodies

# Standard modules
using StaticArrays
using GLMakie
using CUDA
using BasicBSpline
using Interpolations

# Functions for retrieving data from the GPU
function get_omega!(vort,sim)
    @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u) * sim.L / sim.U
    copyto!(vort,sim.flow.σ[inside(sim.flow.σ)])
end
function get_body!(bod,sim,t=WaterLily.time(sim))
    @inside sim.flow.σ[I] = WaterLily.sdf(sim.body,SVector(Tuple(I).-0.5f0),t)
    copyto!(bod,sim.flow.σ[inside(sim.flow.σ)])
end

# Main plotting routine.
function body_omega_fig(sim,resolution=(1400,700))
    # Set up figure
    fig = Figure(;resolution)
    ax = Axis(fig[1, 1]; autolimitaspect=1)
    hidedecorations!(ax); hidespines!(ax)

    # Get first vorticity viz
    vort = sim.flow.σ[inside(sim.flow.σ)] |> Array; ovort = Observable(vort)
    get_omega!(vort,sim); notify(ovort)
    #contourf!(ax, ovort, levels=range(-20, 20, 21), colormap=:bluesreds)
    
    # Set up body viz
    bod = sim.flow.σ[inside(sim.flow.σ)] |> Array; obod = Observable(bod)
    get_body!(bod,sim); notify(obod)
    # Plot the outline of the body.
    #colormap = to_colormap([:grey30,(:grey,0.5)])
    #contourf!(ax,obod,levels=[-100,0,1];colormap)
    # Plot the SDF
    contourf!(ax, obod, levels=range(-5*sim.L, 5*sim.L, 51), colormap=:haline)
    contour!(ax, obod, levels=[0.5])
    
    # Return the objects
    fig, (vort,ovort,bod,obod)
end

# Update data by retrieving values from the GPU.
function update!(viz,sim)
    vort,ovort,bod,obod = viz
    get_omega!(vort,sim); notify(ovort)
    get_body!(bod,sim); notify(obod)
end

# Routine for setting up the simulation.
function make_sim(; L=32, Re=1e3, U=1, n=8, m=4, T=Float32, mem=Array)

    # NOTE: transformations switched off for now.
    nose = SA[L, 0.5f0m*L]
    θ = T(0)

    function map(x, t)
        # Rotation matrix for the constant angle of attack.
        R = SA[cos(θ) -sin(θ); sin(θ) cos(θ)]
        # move to origin and align with x-axis
        ξ = R*(x-nose)
        # Return the transformed coordinate.
#        return ξ
        return SA[ξ[1], abs(ξ[2])]
    end


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
    # TODO values need to be confirmed against the plot from Costello.
    timeVals = [0, 0.01, 0.2, 0.4, 0.95, 1.0]
    
    halfThicknesses = hcat(halfThicknesses[:, 1:1], halfThicknesses, halfThicknesses[:, 1:1], halfThicknesses[:, 1:1])
    lengths = hcat(lengths[:, 1:1], lengths, lengths[:, 1:1], lengths[:, 1:1])
    thetas  = hcat(thetas[:, 1:1], thetas, thetas[:, 1:1], thetas[:, 1:1])
    
    period = 2.0*L
    
    # Moved outside of params_for_profile
    Lfit, thkFit, thetaFit = zeros(size(lengths, 1)), zeros(size(lengths, 1)), zeros(size(lengths, 1))
    
    function jelly(s, t)
        tOverT = (t/period) % period
        
#        Lfit, thkFit, thetaFit = params_for_profile(tOverT, timeVals, lengths, halfThicknesses, thetas)  # TODO this fails
        for i in 1:length(Lfit)
#            itp_l = interpolate(timeVals, lengths[1, :], SteffenMonotonicInterpolation())
#            itp_t = interpolate(timeVals, halfThicknesses[i, :], SteffenMonotonicInterpolation())
#            itp_a = interpolate(timeVals, thetas[i, :], SteffenMonotonicInterpolation())

#            Lfit[i] = itp_l(tOverT)
#            thkFit[i] = itp_t(tOverT)
#            thetaFit[i] = itp_a(tOverT)
        end
        


#        xy, cps, area = profileFromParams(Lfit, thkFit, thetaFit; mirror=true)
#        return evaluate_spline(cps, s)

        return 0.5f0L*SA[1+cos(s*pi), 0.12f0sin(s*pi)]
    end

    body = ParametricBody(jelly, (0, 1); map, T, mem)

    # Baseline - basic ellipse defined using an analytical function
#    ellipse(θ, t) = 0.5f0L*SA[1+cos(θ), 0.12f0sin(θ)]
#    body = ParametricBody(ellipse, (0, π); map, T, mem)
    
    # Test 0.0 - new splines
#=
    s = 0:0.01:1
    cps = hcat(
        rotatePoint([0, -0.2], [0., -0.2], 15.0/180.0*pi),
        rotatePoint([0.15, -0.2], [0., -0.2], 15.0/180.0*pi),
        rotatePoint([0.4, -0.35], [0., -0.2], 15.0/180.0*pi),
        rotatePoint([0.389, -0.52], [0., -0.2], 15.0/180.0*pi),
    )
    pu = evaluate_spline(cps, s)
    xy = hcat(pu, reverse(pu[:, 1:end-1], dims=2))
    a = polyArea(xy)
    
    # Test 0.1 - bell shape params
    smooth_time = 0:0.01:1
    halfThicknesses = hcat(halfThicknesses[:, 1:1], halfThicknesses, halfThicknesses[:, 1:1], halfThicknesses[:, 1:1])
    lengths = hcat(lengths[:, 1:1], lengths, lengths[:, 1:1], lengths[:, 1:1])
    thetas  = hcat(thetas[:, 1:1], thetas, thetas[:, 1:1], thetas[:, 1:1])
    itp_l = interpolate(timeVals, lengths[1, :], SteffenMonotonicInterpolation())
    smooth_time = 0:0.01:1
    y_l = itp_l.(smooth_time)
    
    # Test 0.2 - bell shape
    Lfit, thkFit, thetaFit = params_for_profile(0.35, timeVals, lengths, halfThicknesses, thetas, aTarget=-1.0)
    Lfit_o, thkFit_o, thetaFit_o = params_for_profile(0.35, timeVals, lengths, halfThicknesses, thetas)
    xy, cps, area = profileFromParams(Lfit, thkFit, thetaFit)
    xy_o, cps_o, area_o = profileFromParams(Lfit_o, thkFit_o, thetaFit_o)
=#

    # Test 1 - ellipse-like shape defined using a B-spline
    # NOTE this fails.
#=
    p = 3
    kVec = T[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0] ./ 4.0f0
    controlPoints = Vector{T}[[0.0, 0.0], [0.0, 0.12], [0.2, 0.12], [0.5, 0.13], [0.8, 0.12], [1.0, 0.11], [1.0, 0.0]]
#    kVec ./= maximum(kVec)
    k = KnotVector(kVec)
    P = BSplineSpace{p}(k)
    bell_spline = BSplineManifold(controlPoints, P)
    
    function body_shape(s, t)
        # TODO this fails on the GPU because of type mismatches and voodoo.
         xy = bell_spline(s)
       # TODO can this be done without GPU to CPU data transfer?
#        return L*SA[xy[1], abs(xy[2])]
        
        return 0.5f0*L*SA[1+cos(s*pi), 0.12f0*sin(s*pi)]
    end
    body = ParametricBody(body_shape, (0, 1); map, T, mem) 
=#

    # Test 2 - see if can use Interpolations on the GPU
    # NOTE this fails
#=
    function body_shape(s, t)
        y_points = SA[0.0, 0.1, 0.12, 0.1, 0.0]
        s_vals = range(0, 1, length(y_points))
        s = 0.3f0
        # TODO fails here
#        itp = scale(interpolate(y_points, BSpline(Linear())), s_vals)
#        yi = itp(s)
        
        return 0.5f0*L*SA[1+cos(s*pi), 0.12f0*sin(s*pi)]
    end
    body = ParametricBody(body_shape, (0, 1); map, T, mem) 
=#

    
    
    # Test XX - analytical description of the bell shape.
#=    
    # Define a parametric curve describing the shape
    bellShapeX = s -> sign(s)*(1.0f0-cos(clamp(abs(s), 0, 1)*pi/2.0f0))^0.75f0
    bellShapeY = s -> (sin(pi/2.0f0 - clamp(abs(s), 0, 1)*(pi/2.0f0)))^0.5f0 - 1.0f0
    # Wrap into a single function. Note that the x and y are flipped to align the flow with the x-axis.
    bell(s, t) = 0.5f0L*SA[1.0f0-bellShapeY(s), bellShapeX(s)]

    # Create a ParametricBody to automatically find the nearest point on the curve.
    body = ParametricBody(bell, (0, 1); map, T, mem)
=#
    

    return Simulation((n*L, m*L), (U, 0), L; ν=U*L/Re, body, T, mem)
end

# === main ===
@assert CUDA.functional()

Makie.inline!(false)

L = 32
name = "outputs/out_02_motion.mp4"

cycle = range(0, L*4.0, 51)
sim = make_sim(; L, mem=CuArray)

fig,viz = body_omega_fig(sim)

@time Makie.record(fig, name, cycle, framerate=10) do t
    println("Current time ", t)
    sim_step!(sim, t)
    update!(viz, sim)
end

