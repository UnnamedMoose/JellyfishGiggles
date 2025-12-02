using LinearAlgebra, Printf, Statistics, Plots, ParametricBodies, StaticArrays
using WaterLily, CUDA
using GeometryBasics, Optim
using BiotSavartBCs
using DelimitedFiles, DataFrames
using Interpolations, LinearAlgebra, Dierckx
using CairoMakie

Tp = Float32
T = Float32

include("JellyfishGeometry.jl")

include("InterpolationFunctions.jl")

include("Metrics.jl")

include("SimulationSetup.jl")

include("src/ThreeD_plots.jl")

"""
Current state of numerical model of PROLATE jellyfish:
* Digitised control points from Sahin 2009. Control points deviate as a linear line was added to assure matching curvatures at start/end point of curve. 
* CPs 3 has an adjusted control point to reduce curvature and solve SDF issues.
* A curve is made from the control points, creating half a jellyfish. This curve is resampled into a number of points N. 
* An area optimiser is applied to this new, larger set of control points. It defines a cost function based on area and control points and this is minimised by relocating control points 2 to 10 in order to match the areas of each time step control point set.
* Area is calculated using the shoelace formula, a typically used algorithm for calculating the area of polygons.
* With this new set of control points, the half jellyfish is mirrored around the x-axis in order to acquire a complete jellyfish. 
* The order of the control points is then reversed in order to get a well-defined closed body in WaterLily.
* A linear line is added around (0,0) through adding some extra control points that create a straight line, to acquire a C2-continuous curve.

Current state of simulation setup
* A function is defined for the simulation, currently still called DynamicSpline, where the simulation settings and body are defined.
* The body is based on the NurbsCurve that is generated from the control point sets, curve degree can be changed through deg.
* Domain and body size are dependent on grid size D.
* Simulation is run through the sim_gif_forces function that generates a gif and saves the hydro forces on the body.
* Scaling of the forces is done by dividing the total force through 0.5*D*U^2 but the actual unit of this force is questionable and this should be researched.
* A first try is done towards a moving body, but instead of a real moving body the flow velocity will be variable by making uBC a function.

Tasks:
- Check on influence of changing the control point of set 3 and adding a linear line on simulation reliability.
- Move all variables in the geometry functions outside to the overall function to make input from MAIN easier. Make discretisation number input and such.
- Deeper research into the units that go in and result from WaterLily. 
- Try to acquire a function for the flow velocity uBC that actually represents the relative velocity of the jellyfish.
- Decide on the best options for simulation, a moving body or a differing flow velocity? Discuss with supervisors.
- Forces seem to really get better for D=2^7, but for an even finer grid, I do not know. Computation time is simply too long, how to solve this?
- Need to test the convergence for several settings as well. Think of polynomial degree and such.
- Go into 3D Julia simulation setup. Options for axisymmetry and such.
- Add some sort of velocity reset after each cycle.
- May want to try something with interpNurbs.
- Make body move and consider also the BCs and how these should move (maybe send an e-mail to Marin for an example)
- Convergence studies for flow field, forces and velocities...
- There is a Makie extension in the WL repository that may be useful for visualization.
- Once going to 3D, ask Gabe for 3D parametric bodies and such.

Made a function to plot a pizza slice in 3D to maybe acquire an axisymmetric solution.
A forward moving jellyfish seems a bit weird.
"""

"""
Run the Jellyfish Geometry generator to acquire a control point set list for the simulation.
T.B.A. Scaling by diameter? 
"""

new_cps_list    = create_cps_list(Tp) #.* 2
# itp_x, itp_y = interpolate_points(half_cps_list[3])
""" 
Circle example: Make a generic moving or sinusoidal moving circle 
"""

# new_cps_list    = make_circle_cps(Float32, 22, 10; radius=1.0, shift_per_step=(0.05, 0.02))
new_cps_list    = make_circle_cps_sin_motion(Float64, 22, 10; radius=1.0,freq=0.1, amplitude=(1.0, 1.0))

# """
# Simulation parameters
# D   = number of grid cells over jelly diameter.
# Re  = Reynolds number
# U   = free stream velocity
# ϵ   = thickness ratio (thickness/diameter)
# thk = thickness offset for SDF
# deg = polynomial degree of the NURBS curve
# """

# const v = 0.0f0
# const s = 0.0f0

D = 2^5; Re = 302; U = 1; ϵ = 0.5; thk = 8ϵ; deg = 2; cycles = Tp(1); period = Tp(3); duration = 3                                                     
sim             = TwoDimJellyfish(; new_cps_list, D, Re, U, ϵ, thk, deg, mem=Array, use_biotsavart=true, U_func=U_func)  # mem=CUDA

# """
# Visualisations and Metrics
# """

# # get_shape_error(new_cps_list)                               # Calculate shape error between each control point set
# # generate_grid_view(new_cps_list, D)                         # Generate a view of the grid size relative to the flap size
# get_curves(new_cps_list)                                    # Generate curves of each control point set
# generate_sdf_plots(new_cps_list, thk, D, Tp, deg)           # Generate signed distance function input = (cps_list, thk, grid size, Type, poly degree)
# # plot_interp_shapes(new_cps_list, period, 2.1, 3, 0.1)       # MAX = 10 plots at once    input = (cps_list, t_start, t_end, step)

# for (i, cp) in enumerate(new_cps_list)
#     check_geometry(cp, name="Curve $i")                   # Check curvature of geometry. Might be wrong.
# end

# """
# Run the simulation with GIF output and pressure logging
# """

# WaterLily.logger("test_psolver")
res             = simulate_Jelly_Fourier!(sim, new_cps_list; duration, period, step = 0.1, remeasure = true, plotbody = true)
# plot_logger("test_psolver")
# savefig("psolver.png")

# Choose a representative time
# scaled_cps_list = new_cps_list[1] .* D .+ (D, 2D)
# t = 0.0

# x = range(30, 75; length=30)
# y = range(40, 90; length=30)
# nx, ny = length(x), length(y)

# # Arrays for plotting (only nonzero vectors)
# xs, ys, nxs, nys = Float64[], Float64[], Float64[], Float64[]

# for j in 1:ny, i in 1:nx
#     nvec = WaterLily.nds(sim.body, SVector(x[i], y[j]), 0.0)
#     if norm(nvec) > 1e-6       # skip zero (or near-zero) vectors
#         push!(xs, x[i])
#         push!(ys, y[j])
#         push!(nxs, nvec[1])
#         push!(nys, nvec[2])
#     end
# end

# fig = Figure(resolution=(700,700))
# ax = Axis(fig[1,1], title="Surface Normals (nds)", aspect=DataAspect())

# arrows!(ax, xs, ys, nxs, nys, arrowsize=10, lengthscale=3, color=:blue)

# save("nds_frame_$(round(Int,t)).png", fig)


# interp_cps = []

# plt = Plots.plot(new_cps_list[1][1,:], new_cps_list[1][2,:])
# for (i,t) in enumerate(time)
#     cps = cps_fourier_interpolator(new_cps_list, time,t, Real(3), Int(5))
#     @show cps
#     Plots.plot!(cps[1,:], cps[2,:])
#     push!(interp_cps, cps)
# end
# display(plt)


# D = 2^4; Re = 302; U = 1; ϵ = 0.5; thk = 2(ϵ+√3); deg = 2; cycles = Tp(1); period = Tp(3); duration = cycles * period 
# function compute_r(new_cps_list)
#     cyl_cps_list = []
#     xyz_cps_list = []
#     for cps_set in new_cps_list
#         # @show typeof(cps_set)
#         # cyl_cps = zeros(3, length(axi_cps_list[1][1])) 
#         # @show cps, typeof(cps)
#         x = cps_set[1,:]

#         r = cps_set[2,:]
#         θ = zeros(size(r))  # Angle θ = 0 for the slice in the x-y plane
#         xyz_cps = [x'; (r .* cos.(θ))'; (r .* sin.(θ))']
#         cyl_cps = hcat(x, r, θ)'
#         # @show cyl_cps

#         push!(cyl_cps_list, cyl_cps)
#         push!(xyz_cps_list, SMatrix{3, 105, Float32, 315}(xyz_cps))
#     end
#     return xyz_cps_list
# end

# new_cps_list = compute_r(new_cps_list)



# mirrorto!(a,b) = (a .= b; a)
# #     n = size(b,1)
# #     a[reverse(1:n),reverse(1:n),:].=b
# #     a[reverse(n+1:2n),1:n,:].=a[1:n,1:n,:]
# #     a[:,reverse(n+1:2n),:].=a[:,1:n,:]
# #     return a
# # end

function geom_nomirror!(d, sim, t = WaterLily.time(sim))
    a = sim.flow.σ
    WaterLily.measure_sdf!(a, sim.body, t)
    copyto!(d, a[inside(a)])  # copy SDF data to CPU

    alg = Meshing.MarchingCubes()
    ranges = (1:size(d,1), 1:size(d,2), 1:size(d,3))
    # ranges = range.((0, 0, 0), size(d))  # only use your domain, no mirroring
    points, faces = Meshing.isosurface(d, alg, ranges...)

    p3f = Point3f.(points)
    gltriangles = GLMakie.GLTriangleFace.(faces)
    return GLMakie.normal_mesh(p3f, gltriangles)
end

function ω_nomirror!(d,sim)
    a,dt = sim.flow.σ,sim.L/sim.U
    @inside a[I] = WaterLily.ω_mag(I,sim.flow.u)*dt
    copyto!(d,a[inside(a)]) # copy to CPU
    # mirrorto!(md,d)         # mirror quadrant
end

# GLMakie.activate!()
# # Makie.inline!(false)

# begin
#     # Define geometry and motion
#     sim = ThreeDimJellyfish(; new_cps_list, D, Re, U, ϵ, thk, deg, mem=Array, use_biotsavart=true, U_func=U_func)
#     sim_step!(sim,sim_time(sim)+0.05;remeasure=true);

#     # Create CPU buffer arrays for geometry flow viz 
#     a = sim.flow.σ
#     d = similar(a,size(inside(a))) |> Array; # one quadrant
#     # md = similar(d,(2,2,1).*size(d))  # hold mirrored data

#     # # Set up geometry viz
#     # geom = geom_nomirror!(d,sim) |> Observable;
#     # fig, ax, _ = GLMakie.mesh(geom, color=:red, transparency=false)

#     # #Set up flow viz
#     # ω = ω_nomirror!(d,sim) |> Observable;
#     # volume!(ω, algorithm=:absorption, colormap=:algae, colorrange=(1,10))

#     geom = geom_nomirror!(d, sim) |> Observable
#     ω    = ω_nomirror!(d, sim)    |> Observable

#     fig = Figure()
#     ax  = LScene(fig[1,1], scenekw=(show_axis=false,))

#     volume!(ax, ω; algorithm=:mip, colormap=:algae, colorrange=(1,10))
#     mesh!(ax, geom; alpha=0.1, color=:red)
#     GLMakie.display(fig)
#     fig
# end

# GLMakie.record(fig,"jelly.mp4",1:24;framerate=24) do tᵢ
# # foreach(1:100) do frame
#     # time per frame = step time = 0.1
#     @show tᵢ
#     Tp = eltype(sim.flow.p)
#     t₀ = round(sim_time(sim))
#     t = sum(sim.flow.Δt[1:end-1])
#     period = 24 * sim.L / sim.U

#     total_t = sim_time(sim)

#     while t < tᵢ * sim.L / sim.U  # advance ~0.05 simulated seconds per frame
#         sim.flow.Δt[end] = WaterLily.CFL(sim.flow; Δt_max=Tp(0.1))
#         @show t

#         cps_interp= interpolate_cps_hermite_new(new_cps_list, t, period)
#         body_interpolation = cps_interp .* 1 .* sim.L .+ (Tp(sim.L), Tp(2sim.L), Tp(sim.L/2))
#         body_interpolation = SMatrix{3, 105, Float32, 315}(body_interpolation)

#         sim.sim.body = ParametricBodies.update!(sim.sim.body, body_interpolation, sim.flow.Δt[end])

#         sim_step!(sim,tᵢ ;remeasure=true);
#         t += sim.flow.Δt[end]
#     end

#     geom[] = geom_nomirror!(d,sim);
#     ω[] = ω_nomirror!(d,sim);
# end

# sim             = ThreeDimJellyfish(; new_cps_list, D, Re, U, ϵ, thk, deg, mem=Array, use_biotsavart=true, U_func=U_func)

# res = simulate_Jelly_Makie!(sim, new_cps_list; duration=2, step=0.05, recordfile="Jelly3D.mp4")
# res = simulate_Jelly_Makie!(sim, new_cps_list)