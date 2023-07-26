# WaterLily and misc addons
using WaterLily
using ParametricBodies
# Standard modules
using StaticArrays
using GLMakie
using CUDA
using BasicBSpline
using LinearAlgebra

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
    contourf!(ax, ovort, levels=range(-20, 20, 21), colormap=:bluesreds)
    
    # Set up body viz
    bod = sim.flow.σ[inside(sim.flow.σ)] |> Array; obod = Observable(bod)
    get_body!(bod,sim); notify(obod)
    colormap = to_colormap([:grey30,(:grey,0.5)])
    contourf!(ax,obod,levels=[-100,0,1];colormap)
    
    # Return the objects
    fig,(vort,ovort,bod,obod)
end

# Update data by retrieving values from the GPU.
function update!(viz,sim)
    vort,ovort,bod,obod = viz
    get_omega!(vort,sim); notify(ovort)
    get_body!(bod,sim); notify(obod)
end

# Routine for setting up the simulation.
function make_sim(; L=32, Re=1e3, U=1, n=8, m=4, T=Float32, mem=Array)
    
    function spline_sdf(x, t)

        # Thickness function - symmetric about s=0.5
        thickness = s -> 0.24f0 .* (cos.(abs(s-0.5f0)/0.5f0*pi/2.0f0)).^0.65f0

        # Baseline (undeformed) bell shape
        p = 3
        kVec = Float64[0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4]
        controlPoints = [[-1, -1], [-0.97, -0.55], [-0.2, 0], [0, 0], [0.2, 0], [0.97, -0.55], [1, -1]]

        # Function for rotating points around a specific joint point.
        function rotateSegment(iJoint, iMove, theta)
            for j in iMove
                pt = controlPoints[j] - controlPoints[iJoint]
                xr = pt[1]*cos(theta) - pt[2]*sin(theta)
                yr = pt[2]*cos(theta) + pt[1]*sin(theta)
                controlPoints[j] = [xr, yr] + controlPoints[iJoint]
            end
        end

        # Rotate the control points acoording to the motion.
        # half sine wave so from 0 to 1 and back to 0.
        # NOTE: motion like this does not work with AutoBody...
        period = 1.0  # dummy
        #rotateSegment(5, 6:1:length(controlPoints), 20.0/180.0*pi * sin(pi*t/period))
        #rotateSegment(6, 6:1:length(controlPoints), 25.0/180.0*pi * sin(pi*t/period))
        #rotateSegment(3, 1:2, -20.0/180.0*pi * sin(pi*t/period))
        #rotateSegment(2, 1:1, -25.0/180.0*pi * sin(pi*t/period))

        # Define and instantiate the spline object.
        kVec ./= maximum(kVec)
        k = KnotVector(kVec)
        P = BSplineSpace{p}(k)
        bell_spline = BSplineManifold([[-v[2], v[1]] for v in controlPoints], P)

        # Define curve function returning a 2D point based on the parameter `s`
        curve_function = s -> bell_spline.(s)[1:2]

        # Initialize the minimum distance and closest point
        min_distance = Inf

        # Will look for relative change in s for the nearest point (according to SDF)
        # to be below a theshold level.
        s_old = 10.
        s_min = 0.5

        # Search for minimum SDF value using a reduced-trust region approach.
        # This is adopted to allow for complex spline shapes with multiple local
        # minima, but is expensive. Maybe some simple optimiser like a Golden rule
        # search of Newton-Raphson could work better for simple shapes?
        iter = 1
        while abs(s_old-s_min) > 0.01
            if iter == 1
                s_old = 10.
            else
                s_old = s_min
            end

            # NOTE: for jagged sdf contours, push the no. points up.
            s = range(clamp(s_min-0.5/iter, 0, 1), clamp(s_min+0.5/iter, 0, 1), 21)

            # Iterate over target points and find the closest point
            # TODO is there an argmin in Julia? This could become a one-liner
            for i in 1:1:length(s)
                distance = norm(x - (curve_function(s[i])*L + [L, 2*L])) - thickness(s[i])*L
                if distance < min_distance
                    min_distance = distance
                    s_min = s[i]
                end
            end

            # Safety.
            iter += 1
            if iter > 10
                break
            end
        end

        return min_distance
    end

	# make the fish simulation
	body = AutoBody(spline_sdf)
    return Simulation((n*L, m*L), (U, 0), L; ν=U*L/Re, body, T, mem)
end

# === main ===
@assert CUDA.functional()

Makie.inline!(false)

L = 32
name = "outputs/out_01_motion.mp4"

cycle = range(0, L*2, 31)
# NOTE: SDF is not compatible with GPU computations :*(
sim = make_sim(; L)#, mem=CuArray)
fig,viz = body_omega_fig(sim)

@time Makie.record(fig, name, cycle, framerate=10) do t
    println("Current time ", t)
    sim_step!(sim, t)
    update!(viz, sim)
end

