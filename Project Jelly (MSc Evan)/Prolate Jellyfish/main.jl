using LinearAlgebra, Printf, Statistics, Plots, ParametricBodies, StaticArrays
using WaterLily, CUDA       ### Note that WaterLily functionalities have been adjusted before moving to DelftBlue.
using GeometryBasics, Optim
using BiotSavartBCs
using Interpolations: LinearInterpolation
using DelimitedFiles, DataFrames, CSV
using GLMakie
using Dierckx
using Images, ImageMagick, ImageIO
using Meshing

import WaterLily: @loop,scale_u!,conv_diff!,udf!,accelerate!,BDIM!

import WaterLily: CFL
CFL(a::Flow;Δt_max=10) = 0.1

import WaterLily: sim_step!
function sim_step!(sim::AbstractSimulation,t_end;remeasure=true,λ=quick,max_steps=typemax(Int),verbose=false,
        udf=nothing,kwargs...)
    steps₀ = length(sim.flow.Δt)
    while sim_time(sim) < t_end && length(sim.flow.Δt) - steps₀ < max_steps
        sim_step!(sim; remeasure, λ, udf, kwargs...)
        verbose && sim_info(sim)
    end
end

import BiotSavartBCs: biot_mom_step!,biot_project!
function biot_mom_step!(a::Flow{N},b,ω...;λ=quick,udf=nothing,fmm=true,U,kwargs...) where N
    a.u⁰ .= a.u; scale_u!(a,0); t₁ = sum(a.Δt); t₀ = t₁-a.Δt[end]
    # predictor u → u'
    @log "p"
    conv_diff!(a.f,a.u⁰,a.σ,λ,ν=a.ν)
    udf!(a,udf,t₀; kwargs...)
    BDIM!(a);
    biot_project!(a,b,ω...,U;fmm) # new
    # corrector u → u¹
    @log "c"
    conv_diff!(a.f,a.u,a.σ,λ,ν=a.ν)
    udf!(a,udf,t₁; kwargs...)
    BDIM!(a); scale_u!(a,0.5)
    biot_project!(a,b,ω...,U;fmm,w=0.5) # new
    push!(a.Δt,CFL(a))
end

Tp = Float64; T = Float64

"""
Include the background packages where functions are formed.
"""

include("Background_functions.jl")

""" --- Simulation Parameters --- """
ThreeD      = true             # ThreeD optionality
D           = 2^5               # Grid size of Jellyfish diameter
Re          = 302               # Reynolds Number (From Sahin 2009) (UD/ν) based on avg medusa velocity of 2.42 cm/s
St          = 0.52              # Strouhal number (From Sahin 2009) (D/(UT)) based on avg medusa velocity of 2.42 cm/s
U           = 1                 # Reference velocity
ϵ           = 1                 # Boundary cell thickness
thk         = 2ϵ+√3             # Boundary layer thickness
deg         = 2                 # Polynomial degree to describe jellyfish boundary
cycles      = 5                 # user-defined number of motion cycles
period      = (D/U) / St        # jellyfish motion period is ~1 second, non-dimensionalised
duration    = cycles * period   # total duration of motion

""" --- Control Point Generation ---
Generation Process:
1. A set of control points, digitized from Sahin 2009, representing the bell kinematics of the (half) jellyfish, is used to define the motion of the body.
2. This set of control points is split into a contraction and expansion phase, consisting of 5 steps each, with the starting cps set of the next phase added at the end of the current phase to ensure continuity.
3. The number of control points to define each time step is extended to a number of Ncps (first input of constructor).
4. Next, the number of frames to define a full cycle is expanded, using a spline method (or the exponential smoothing method).
    This spline method defines 2 '1D splines' for each control point, to describe its motion in x- and y-direction as a spline.
    From this spline, it derives a new and extended pathing for each contorl point coordinate. The spline should ensure continuity of the pathing, as it is a parametrisation.
    To define smoothing, adjust the spline_s in the constructor. The number of frames to define a full cycle is contorlled by the upsample variable in the constructor. # upsamples is added in between each original frame.
5. The new set of frames with control points is then mirrorred and the control point order reversed to acquire the right definition of the body for WaterLily.jl
6. At the origin of the jellyfish, points are added to create C2 continuity. With the points, basically a straight line is formed at the origin.
7. It outputs the paths of each coordinate (for x and y), the contraction and expansion control point sets and the full cps list.

Right now I hardcoded the expansion phase to be 2* the contraction phase.
Input Parameters:
construct_jelly_motion: Ncps = number of control points, spline_s = spline smoothing factor, up = number of upsamples.
γ is the fraction between the length of the expansion phase and the contraction phase
λ_area is the strictness of area conservation for the conservation optimiser
λ_shape is the strictness of keeping shape during optimising.
α is the exponential smoothing coefficient

What influences pathing (smoothness)?: spline smoothing, number of upsamples, exponential smoothing, are conservation optimiser
So check 'convergence' of spline_s, up, α for control point pathing.

In terms of shape and sdf, the Ncps and deg are important parameters to consider.

γ influences the kinematics, it will be a control parameter for the actual research.
"""
Ncps = 50; spline_s = 0.001; up = 10; α=0.250

path_x, path_y                                      = construct_jelly_motion(Ncps,spline_s,up, deg)
path_x                                              = [blend_cycles(p, 35) for p in path_x]
path_y                                              = [blend_cycles(p, 35) for p in path_y]
len                                                 = length(path_x)
path_x_smooth                                       = [exp_smooth(path_x[i], α) for i in 1:len]
path_y_smooth                                       = [exp_smooth(path_y[i], α) for i in 1:len]  
frame_points                                        = range(1,length(path_x_smooth[25]), step=1)
pathing                                             = control_point_functions(path_x_smooth, path_y_smooth, frame_points)

""" --- Control Point Generation Debugging ---
Different plotting routines to check if all functions are well-behaved and the results create continuous CPS motion.
Particularly for control point 33, which is the outer point of the jellyfish flap is good for checks.
Function to display a plot of the actual jelly shape at a specific frame. 
140 frames equals 1 period -> Should be changed to numerical time.
140 / ~60 (dimensionless period) -> dimensionless time?
"""
frame_check     =   100
cp_check        =   33
# display(Plots.plot(cps_list_new[frame_check][1,:], cps_list_new[frame_check][2,:], xlabel="x-coordinate", ylabel="y-coordinate", title="Jellyfish shape frame $frame_check"))
display(Plots.plot(pathing[cp_check](collect(0:1:500))[1], xlabel="frame number", ylabel="x-coordinate", title="x-pathing of CP33")) # control point pathing
# display(Plots.plot!(pathing[cp_check](0)[1] .+ accumulate(+,[vel_chan[cp_check](i)[1] for i in 0:1:500]),xlabel="frame number", ylabel="x-coordinate", title="x-velocity of CP$cp_check" )) # control point velocity
# display(Plots.plot([get_area(cps_at_time(pathing, 105, t; ThreeD=false) .* D) for t in 1:1:500]/get_area(cps_at_time(pathing,105,1) .* D), xlabel="frame number", ylabel="Relative Area", title="Area Relative to Initial Area"))

""" --- Simulation Setup 1 ---
1. Define the starting location for the jellyfish.
2. Setup the simulation 'environment' for the jellyfish. Requires input from the simulation parameters above.
3. Viscosity is calculated from velocity and length scale and the Reynolds number.
4. The body is constructed using the DynamicNurbsBody from the ParametricBodies package.
5. The knots array is automatically generated from the polynomial degree and number of control points.
6. Weights are all put to 1 for all control points.
7. Can be setup with either the BiotSavartBCs or a general WaterLily simulation.
8. Domain size and inflow velocity can be adjusted in here. --> Might want to move this to input parameters in some way.

Convergence studies regarding the following simulation parameters should be done:
Naturally, D = 2ᵖ
Lets keep Re and U constant.
ϵ and thk should definitely be checked for its influence on the forces.
"""

cps_start = cps_at_time(pathing, 105, 0; ThreeD=false) # defined from t = 0 to t = 545, which are actually frames.

@inline function TwoDimJellyfish(::Type{T}=Float32; new_cps_list, D=2^7, Re=302, U=1, ϵ=0.5, thk=2ϵ+√3, deg, mem=Array, use_biotsavart=false) where {T<:AbstractFloat}
    ν           =   U * D / Re

    cps         =   new_cps_list .* 1 .* D .+ SA{T}[2D, 2.5D]
    degree      =   deg
    n_ctrl      =   size(cps, 2)
    weights     =   ones(T, n_ctrl)
    knots       =   T.(clamped_uniform_knots(degree, n_ctrl))
    curve       =   NurbsCurve(cps, knots, weights)
    body        =   DynamicNurbsBody(curve; thk=thk, boundary=true)

    return use_biotsavart ? BiotSimulation((6D, 6D), (0,0), D; U, ν, body, T, mem, ϵ) : Simulation((6D, 6D), (0,0), D; U, ν, body, T, mem, ϵ)
end

""" --- Simulation Setup 2 ---
In this part the actual simulation can be run. 
1. Input for this is all defined above using the simulation parameters. The starting position of the jellyfish is defined in the above section from cps_start.
2. The body is updated each timestep using the ParametricBodies update functionality
3. The sim / flow is updated with sim_step.
4. Pressure solver statistics can be put on to check the performance of WaterLily.

Currently this is able to compute force, acceleration, velocity, displacement, pressure plots and vorticity plots/gif.
All plotting routines for the above parameters are currently set to true, but can be commented to remove them. --> Add an optionality to put each parameter plot either on or off.

Then there is also the possibility to write all output into a CSV-file for easier computation.
"""
# plt = Plots.plot()
# for t in [54.95, 55.05, 55.15, 55.20, 55.25, 55.30, 56]
#     Plots.plot!(cps_at_time(pathing, 105, t)[1,:],cps_at_time(pathing, 105, t)[2,:], xlims=(1,1.25), ylims=(0.25,0.50))
# end
# display(plt)

"""
Added Mass from MCHenry 2003: Hemiellipsoidal approach
A = α * ρ * Vₛ * acc
α = (2h/d)^(1.4)        Added mass coefficient, h and d are the bell shape parameters.
Vₛ = π * d^2 * hₛ / 6    hₛ is the cavity height, d is cavity diameter, both supposedly a function of time.
For Sarsia sp. the max diameter is 1.25 cm during expansion. 1.15 cm during contraction.
Height is 1.20 cm during the expansion phase. 1.40 cm during contraction.
Cavity volume is very variable, average at about 0.80 cm.
"""

# falling body acceleration term
fall!(flow,t;acceleration) = for i ∈ 1:ndims(flow.p)
    WaterLily.@loop flow.f[I,i] += acceleration[i] over I ∈ CartesianIndices(flow.p)
end

function run_jelly_simulation(cps_start, D, Re, U, ϵ, thk, deg, pathing)
    sim         = TwoDimJellyfish(; new_cps_list=cps_start, D, Re, U, ϵ, thk, deg, mem=Array, use_biotsavart=true)
    forces      = []; forces_filt = []; forces_out = []; time = []; time_sim = []; timesteps = []; displacement = []; velocity = []; acceleration = []
    n_cps       = length(cps_start)
    cps_paths_x = [[] for _ in 1:n_cps]
    prev_force  = 0
    duration    = 25; t₀ = round(sim_time(sim)); step = 0.1
    t0 = 0; a0 = 0; v0 = 0; p0 = 0; Area = get_area(cps_start .* sim.L)
    hₛ = 0.85*D; dₛ=0.8*D; d=1.2*D; h=1.3*D
    mₐ = (2*h / d)^1.4 * (π * dₛ^2 * hₛ) / (6)

    for tᵢ in range(t₀, t₀ + duration; step)        
        t = sum(sim.flow.Δt[1:end-1])
        while t < tᵢ * sim.L / sim.U
            cps             = cps_at_time(pathing, 105, t) .* D .+ SA{T}[2D, 2.5D]

            sim.sim.body    = ParametricBodies.update!(sim.sim.body, cps, sim.flow.Δt[end])
            # sim_step!(sim, t/sim.L; remeasure = true)

            measure!(sim)
            biot_mom_step!(sim.flow,sim.pois,sim.ω,sim.x₀,sim.tar,sim.ftar;
                           fmm=sim.fmm,udf=fall!,acceleration=SA[-a0,0.f0],U=SA[-v0,0.0]) # change of frame
            
            force           =   -WaterLily.pressure_force(sim)[1]
            filt_force      =   0.1 * force + (1-0.1) * prev_force
            Δt              =   sim.flow.Δt[end]
            accel           =   (filt_force + mₐ * a0) / (Area + mₐ)
            p0              +=  Δt * (v0 + Δt * accel / 2.)
            v0              +=  Δt * accel
            a0              =   accel
            @show t, force

            push!(velocity, v0)
            push!(displacement, p0)
            push!(acceleration, a0)
            push!(timesteps, sim.flow.Δt[end])
            push!(time, t * sim.U / sim.L)
            push!(forces, force)
            push!(forces_filt, filt_force)
            push!(time_sim, sim_time(sim))
            for (i, p) in enumerate(cps[1, :])
                push!(cps_paths_x[i], p)
            end

            t0 = t; t += sim.flow.Δt[end]; prev_force = filt_force
        end

        force_out   =   -WaterLily.pressure_force(sim) / (sim.L * 0.5)
        R           =   inside(sim.flow.p)

        gen_p_plots(sim, tᵢ)
        # gen_n_plots(sim, tᵢ)
        gen_ω_gif(sim, tᵢ, R)
        push!(forces_out, force_out[1]) # plot
        println("tU/L=", round(tᵢ, digits = 4), ", Δt=", round(sim.flow.Δt[end], digits = 3))
    end 

    # comp_force = Area .* diff(v0) ./ sim.flow.Δt
    display(Plots.plot(time, acceleration, xlabel="tU/L", ylabel="acceleration", title="Acceleration of Jellyfish"))
    display(Plots.plot(cps_paths_x[33], xlabel= "tU/L", ylabel= "displacement", title="True displacement CP33"))
    display(Plots.plot(time, forces, xlabel="tU/L", ylabel="force", title="Pressure Force on Jellyfish"))
    display(Plots.plot(time, cumsum(forces),xlabel="tU/L", ylabel="force",title="Jellyfish Cumulative Force"))
    display(Plots.plot(time, displacement,xlabel="tU/L", ylabel="displacement",title="Jellyfish Displacement"))
    display(Plots.plot(time, velocity,xlabel="tU/L", ylabel="velocity",title="Jellyfish Velocity"))
    # display(Plots.plot(get_area(cps_start.*D) * diff(velocity)./ timesteps[2:end], xlabel= "timesteps",ylabel="Force", title="Force from F=ma"))

    return forces, forces_out, forces_filt, time, time_sim, timesteps, cps_paths_x, displacement, velocity, acceleration
end

WaterLily.logger("test_psolver")
forces, force_out, force_filt, time, time_sim, timesteps, cps_paths_x, displacement, velocity, acceleration = run_jelly_simulation(cps_start, D, Re, U, ϵ, thk, deg, pathing)
plot_logger("test_psolver")
savefig("psolver.png")

""" --- Simulation results bookkeeping --- 
For some reason, the GIF maker functions decided not to work anymore... They partly do now
"""

# open("results.csv", "w") do io
#     println(io, "forces,time,time_sim,timesteps,displacement,velocity,acceleration")
#     n = length(forces)
#     for i in 1:n
#         f       = forces[i]
#         tnum    = time[i]
#         tsim    = time_sim[i]
#         tsteps  = timesteps[i]
#         dis     = displacement[i]
#         vel     = velocity[i]
#         acc     = acceleration[i]
#         println(io, "$f,$tnum,$tsim,$tsteps,$dis,$vel,$acc")
#     end
# end

# create_gif_from_folder("Prolate Jellyfish/Normals_check/", "Prolate Jellyfish/normals_output.gif", delay=0.05)
# create_gif_from_folder("Prolate Jellyfish/Pressure_check/", "Prolate Jellyfish/pressure_output.gif", delay=0.05)
create_gif_from_folder("Prolate Jellyfish/Vorticity_check/", "Prolate Jellyfish/vorticity_output.gif", delay=0.05)
# create_gif_from_folder("Pressure_check/", "pressure_output.gif", delay=0.05)

""" --- Making the Jellyfish Move Forward ---
0st try, using generated velocity on body boundary with a Ufunc.
1st try, directly updating the offset .+ SA{T}[2D, 2.5D] with its new position: .+ SA{T}[2D + p0, 2.5D]
    Forces blew up to 10^13 so had to stop the simulation.
    Possible reasons for blow up:
        - Direct motion through offset change simply not viable --> Try moving the static jelly forward in this way (I think that worked??)
        - Motion becomes simply too large at once for the solver to be able to handle it --> Split the simulation into a first simulation to define forward velocity and displacement, second one to actually make the jelly move.
2nd try, using the statically defined velocity/position change and implementing this into a new simulation
"""

function get_kinematics()
    kinematic_df = CSV.read("results.csv", DataFrame)
    # kinematic_df = [forces, numerical time, simulation time, time steps, displacement, velocity, acceleration]
    p = kinematic_df[:, 5]
    v = kinematic_df[:, 6]
    a = kinematic_df[:, 7]
    return (p=p, v=v, a=a)
end

# kinematics = get_kinematics()

@inline function MovingJellyfish(kinematics, ::Type{T}=Float32; new_cps_list, D=2^7, Re=302, U=1, ϵ=0.5, thk=2ϵ+√3, deg, mem=Array, use_biotsavart=false) where {T<:AbstractFloat}

    motion_map(x,t) = SA[x[1] + kinematics.p[t], x[2]] ## When sims go out of phase this will definitely fuck up

    cps             = new_cps_list[1] .* D/2 
    degree          = deg
    n_ctrl          = size(cps, 2)
    weights         = ones(T, n_ctrl)
    knots           = T.(clamped_uniform_knots(degree, n_ctrl))
    curve           = NurbsCurve(cps, knots, weights)

    body            = ParametricBody(curve;map=motion_map,ndims=2)

    ν               = U * D / Re

    return use_biotsavart ? BiotSimulation((6D, 6D), (0,0), D; U, ν, body, T, mem, ϵ) : Simulation((6D, 6D), (0,0), D; U, ν, body, T, mem, ϵ)
end

""" --- Validation of the 2D model ---
Use the kinematics in Jellyfish_Kinematics.xlsx to validate the results of the 2D WaterLily model.
"""

# function jelly_sdf(x, t) ## Option, but not differentiable for WaterLily
#     D = 2^5; Re = 302; U = 1; ϵ = 0.75; thk = 2ϵ+√3; deg = 2; n_ctrl = 105
#     cps = cps_at_time(pathing, n_ctrl, t)
#     weights = ones(Tp, n_ctrl)
#     knots = Tp.(clamped_uniform_knots(deg, n_ctrl))
#     curve = NurbsCurve(cps .* (D) .+ 2*D, knots, weights)
#     body  = DynamicNurbsBody(curve; thk=thk, boundary=true)
#     return sdf(body, x, t)
# end

# jelly_map(x,t) = x
# jelly_shape = AutoBody(jelly_sdf, jelly_map)

# # # Make grid
# # xs = range(-6.25, 6.25, length=200)
# # ys = range(-6.25, 6.25, length=200)
# # X, Y = [x for x in xs, y in ys], [y for x in xs, y in ys]  # coordinate mesh

# # # Choose times you want to visualise
# # times = [0,30]

# # plt = Plots.plot(; aspect_ratio=:equal, legend=:topright,
# #            title="Jellyfish geometry evolution", xlabel="x", ylabel="y")
# # for t in times
# #     # Compute SDF field at this t
# #     Z = [ϕ(SA{Tp}[x,y], t) for y in ys, x in xs]

# #     # Plot φ=0 contour (the jellyfish boundary)
# #     Plots.contour!(xs, ys, Z; levels=[0.0], label="t = $(round(t,digits=2))")
# # end
# # display(plt)

""" --- A 3D Jellyfish ---
Use a mapping function to revolve the 2D jellyfish around its axis of symmetry.
"""

function make_3D_Jellyfish(cps_start, D, Re, U, deg, ϵ, T)
    rev_map(x,t)    = SA[(x[1]), hypot(x[2], x[3]) ] 
    cps_j           = cps_start  .* D .+ SA_F32[0.5D; 0]
    degree          = deg
    n_ctrl          = size(cps_j, 2)
    weights_j       = ones(T, n_ctrl)
    knots_j         = T.(clamped_uniform_knots(degree, n_ctrl))
    curve_j         = NurbsCurve(cps_j, knots_j, weights_j)
    body            = ParametricBody(curve_j; map=rev_map, ndims=3)
    ν               = U * D / Re
    sim             = BiotSimulation((3D, D, D), (0,0,0), D; U, ν, body, T, mem=Array, ϵ)
end

sim                 = make_3D_Jellyfish(cps_start, D, Re, U, deg, ϵ, T)
visualize_sdf_3D(sim.body; D=D, n=50, surface_only=false)

Makie.inline!(false)

begin
    # Define geometry and motion on GPU
    sim             = make_3D_Jellyfish(cps_start, D, Re, U, deg, ϵ, T)#mem=CUDA.CuArray);
    cps             = cps_at_time(pathing, 105, 0.05) .* D .+ SA_F32[0.5D; 0]
    sim.sim.body    = ParametricBodies.update!(sim.sim.body, cps, sim.flow.Δt[end])
    
    sim_step!(sim,sim_time(sim)+0.05; remeasure = true);
    @show sim_time(sim)

    # Create CPU buffer arrays for geometry flow viz 
    a               = sim.flow.σ
    d               = similar(a,size(inside(a))) |> Array; # one quadrant
    md              = similar(d, (1,2,2).*size(d))

    # Set up geometry viz
    geom            = geom!(md,d,sim) |> Observable;
    ω               = ω!(md, d, sim) |> Observable

    fig             = GLMakie.Figure()
    ax              = GLMakie.Axis3(fig[1, 1], aspect = :data)

    GLMakie.mesh!(ax, geom, alpha=0.1, color=:red)
    GLMakie.volume!(ax, ω;algorithm=:mip,transparency=true,alpha=0.5,colormap=:algae,colorrange=(1,10))

    fig
end

GLMakie.record(fig,"3D_moving_jelly.mp4",1:25; framerate=5) do frame
    cps             = cps_at_time(pathing, 105, frame) .* D .+ SA_F32[0.5D; 0]
    sim.sim.body    = ParametricBodies.update!(sim.sim.body, cps, sim.flow.Δt[end])
    
    @show frame

    sim_step!(sim,sim_time(sim)+0.05; remeasure = true);
    geom[]          = geom!(md,d,sim);
    ω[]             = ω!(md,d,sim);
end



