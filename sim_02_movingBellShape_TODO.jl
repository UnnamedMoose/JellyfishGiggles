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


    # Test 0 - basic ellipse defined using an analytical function
#=
    ellipse(θ, t) = 0.5f0L*SA[1+cos(θ), 0.12f0sin(θ)]
    body = ParametricBody(ellipse, (0, π); map, T, mem) 
=#

    # Test 1 - ellipse-like shape defined using a B-spline
    # NOTE this fails.
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

cycle = range(0, L*0.1, 2)
sim = make_sim(; L, mem=CuArray)

fig,viz = body_omega_fig(sim)

@time Makie.record(fig, name, cycle, framerate=10) do t
    println("Current time ", t)
    sim_step!(sim, t)
    update!(viz, sim)
end

