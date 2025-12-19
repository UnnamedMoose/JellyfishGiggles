using LinearAlgebra, Printf, Statistics, Plots, ParametricBodies, StaticArrays
using WaterLily, CUDA
using GeometryBasics, Optim
using BiotSavartBCs
using DelimitedFiles, DataFrames

Tp = Float32

include("JellyfishGeometry.jl")

include("InterpolationFunctions.jl")

include("Metrics.jl")

include("SimulationSetup.jl")

"""
Run the Jellyfish Geometry generator to acquire a control point set list for the simulation.
T.B.A. Scaling by diameter? 
"""

new_cps_list    = create_cps_list(Tp) #.* 2

""" 
Circle example: Make a generic moving or sinusoidal moving circle 
"""

# new_cps_list    = make_circle_cps(Float32, 22, 10; radius=1.0, shift_per_step=(0.05, 0.02))
# new_cps_list    = make_circle_cps_sin_motion(Float32, 22, 10; radius=1.0,freq=0.1, amplitude=(1.0, 1.0))

"""
Simulation parameters
D   = number of grid cells over jelly diameter.
Re  = Reynolds number
U   = free stream velocity
ϵ   = thickness ratio (thickness/diameter)
thk = thickness offset for SDF
deg = polynomial degree of the NURBS curve
"""

#U_func(x, t, L) = SA[0.025f0 * t, 0f0]
D = 2^5; Re = 302; U = 1; ϵ = 0.5; thk = 1; deg = 2; cycles = Tp(1); period = Tp(3); duration = cycles * period                                                     
sim             = dynamicSpline(; new_cps_list, D, Re, U, ϵ, thk, deg, mem=Array, use_biotsavart=true)#, U_func=U_func)  # mem=CUDA

"""
Visualisations and Metrics
"""

# get_shape_error(new_cps_list)                               # Calculate shape error between each control point set
# generate_grid_view(new_cps_list, D)                         # Generate a view of the grid size relative to the flap size
# get_curves(new_cps_list)                                    # Generate curves of each control point set
# generate_sdf_plots(new_cps_list, thk, D, Tp, deg)           # Generate signed distance function input = (cps_list, thk, grid size, Type, poly degree)
# plot_interp_shapes(new_cps_list, period, 2.1, 3, 0.1)       # MAX = 10 plots at once    input = (cps_list, t_start, t_end, step)

# for (i, cp) in enumerate(new_cps_list)
#     check_geometry(cp, name="Curve $i")                   # Check curvature of geometry. Might be wrong.
# end

"""
Run the simulation with GIF output and pressure logging
"""

WaterLily.logger("test_psolver")
res             = sim_gif_forces!(sim, new_cps_list; duration, period, step = 0.1, remeasure = true, plotbody = true)
plot_logger("test_psolver")
savefig("psolver.png")

