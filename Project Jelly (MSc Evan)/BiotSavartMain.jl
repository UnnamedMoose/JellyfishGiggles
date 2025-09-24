using LinearAlgebra, Printf, Statistics, Plots, ParametricBodies, StaticArrays
using WaterLily, CUDA
using GeometryBasics, Optim
using BiotSavartBCs
using DelimitedFiles, DataFrames

include("JellyfishGeometry.jl")

include("Metrics.jl")

include("SimulationSetup.jl")

## Generate the control points for each timestep and scale them with the maximum diameter (1.25 cm) to get dimensionless coordinates. 
new_cps_list    = create_cps_list(Float32) ./ Float32(1.25)
# get_cps_scatter(new_cps_list)                                                                         # Get_scatter to generate plots.

## Circle example
### Make a sinusoidal moving circle
# new_cps_list    = make_circle_cps(Float32, 22, 10; radius=1.0, shift_per_step=(0.05, 0.02))

# new_cps_list    = make_circle_cps_sin_motion(Float32, 22, 10; radius=1.0, shift_per_step=(0.05, 0.02),freq=0.1, amplitude=(1.0, 1.0))

## Control point optimisation by area matching and shape preservation
s_vals          = range(0, stop=1, length=500)                                                          # Sample points 
ref_crv         = BSplineCurve(new_cps_list[1]; degree=2)                                               # Reference curve for area comparison
ref_points      = [ref_crv(s) for s in s_vals]                                                          # Evaluate the reference curve at the sampled points
reference_area  = poly_area(ref_points)                                                                 # Calculate the area of the reference polygon
new_cps_list    = [optimize_control_points(cps, reference_area) for cps in new_cps_list]
# new_cps_list    = Vector{SMatrix{2,41,Float32,82}}(new_cps_list)
crvs            = [BSplineCurve(cps; degree=2) for cps in new_cps_list]                                 # Create a list of B-spline curves from control points
# get_curves(crvs)                                                                                      # Plot the curves       
diff = new_cps_list[1] - new_cps_list[end]
# diff = compute_diffs(new_cps_list)

## Calculate areas and relative errors to check optimisation
points          = [[curve(s) for s in s_vals] for curve in crvs]                                        # Evaluate each curve at the sampled points
areas           = [poly_area(pts) for pts in points]                                                    # Calculate the area of each polygon defined by the points
rel_area        = [area / areas[1] for area in areas]                                                   # Relative area compared to the first shape        
rel_errors      = [(area - areas[1]) / areas[1] * 100 for area in areas]                                # Relative area error in percentage
# get_shape_error(areas, rel_errors, rel_area)                                                          # Plot the area and relative error table

### Check the divergence, make a GIF of this
## Setup the simulation with the generated control points
D = 2^6; Re = 302; U = 1; ϵ = 0.5; thk = 2ϵ + √3                                                        # Simulation parameters, D = number of grid cells over jelly diameter.
sim             = dynamicSpline(; new_cps_list, D, Re, U, ϵ, thk, mem=CuArray, use_biotsavart=true);
Tp              = eltype(sim.flow.p) 
periodic_force  = Tp(0); v = Tp(0); s = Tp(0); areas = Tp[]; τ_locals = Tp[]

include("InterpolationFunctions.jl")

period          = Tp(3) * sim.L / sim.U        # period in simulation units
Δt_interp       = 0.05 * sim.L / sim.U      # spacing between interpolated frames
time            = 0:Δt_interp:period            # 5 periods total
interp_cps_cycle = SMatrix{2,41,Float32,82}[]  # store precomputed cps
# interp_cps_cycle = []

yrange          = 15:1:64 #-D/2:1:D/2
xrange          = 100:1:150 #0:1:1.25*D

### Plot grid cells behind jelly for different sizes.
plt             = plot(aspect_ratio=1, xlabel="x", ylabel="y", xlims=(100,150), ylims=(15,64),
           title="Interpolated Shapes", legend=false)
           
for x in xrange
    plot!([x, x], [first(yrange), last(yrange)],
          color=:gray, alpha=0.5, linewidth=0.5)
end

for y in yrange
    plot!([first(xrange), last(xrange)], [y, y],
          color=:gray, alpha=0.5, linewidth=0.5)
end

for t in time
  interpolated  = interpolate_cps_hermite_new(new_cps_list, Tp(t), period)
  interpolated  = SMatrix{2,41,Float32,82}(interpolated)
  push!(interp_cps_cycle, interpolated)
  plot!(interpolated[1,:] * sim.L, interpolated[2,:] * sim.L;
        label="t=$(round(t/sim.L,digits=2))")
end

display(plt)
savefig("grid_size.png")

cycles          = 1
interp_cps_list = vcat([interp_cps_cycle[1:end] for _ in 1:cycles]...)  
duration        = cycles * period / (sim.L / sim.U)

# index           = Int(3 / 0.05 + 1)
# diff            = interp_cps_list[index + 1] - interp_cps_list[index]
# diff2           = interp_cps_list[index] - interp_cps_list[index - 1]
# @show diff, diff2

# scatter(interp_cps_list[1][1,:], interp_cps_list[1][2,:], label="cps_0", aspect_ratio=1)
# scatter!(interp_cps_list[end-3][1,:], interp_cps_list[end-3][2,:], label="cps_9")

# res             = get_forces!(sim; duration=3.5,step=0.1,verbose=true)

# avg_cps_change(new_cps_list)

WaterLily.logger("test_psolver")

res             = sim_gif_forces!(sim, interp_cps_list; duration, step = 0.1, remeasure = true, plotbody = true)

# show the pressure logger
plot_logger("test_psolver")
savefig("psolver.png")

data = readdlm("test_psolver.log", ',', skipstart=1, String)

# Convert to DataFrame for easier handling
df = DataFrame(pc = data[:,1],
               iter = parse.(Int, data[:,2]),
               rinf = parse.(Float64, data[:,3]),
               r2   = parse.(Float64, data[:,4]))

# Forward-fill missing pc values (blank entries "")
for i in 2:nrow(df)
    if df.pc[i] == ""
        df.pc[i] = df.pc[i-1]
    end
end

# Now split predictor vs corrector
df_pred = filter(:pc => ==("p"), df)
df_corr = filter(:pc => ==("c"), df)

# Residuals (log scale is typical for residuals)
# Left y-axis = forces
forceplt = plot(res.ts[1:end], res.forces[1:end];
     label="Force",
     xlabel="tU/L",
     ylabel="Non-dim. Force",
     color=:red,
     xgrid=true,
     gridstyle=:dash,
     gridalpha=0.7)

predinfplt = plot(df_pred.rinf[17:end];
      yscale=:log10,
      label="L∞ predictor",
      color=:blue,
      xlabel = "time step",
      ylabel="Residual")
      plot!(twiny(), df_corr.rinf[17:end];
      label="L∞ corrector",
      color=:red)
# corrinfplt = plot(df_corr.rinf[17:end];
#       yscale=:log10,
#       label="L∞ corrector",
#       color=:red,
#       xlabel = "step",
#       ylabel = "residual")

prediterplt = plot(df_pred.iter[1:end];
      label = "It. predictor",
      xlabel = "time step",
      ylabel = "Iterations",
      color =:blue)
      plot!(twiny(), df_pred.iter[1:end];
      label = "It. corrector",
      color =:red)
# corriterplt = plot(df_corr.iter[1:end];
#       label="It. corrector",
#       xlabel = "time step",
#       ylabel = "Iterations",      
#       color=:red)

savefig(forceplt, "forces.png")
savefig(predinfplt, "Linf.png")
savefig(prediterplt, "Iterations.png")

# plot(rinf_clipped, label="Predictor r∞", yscale=:log10, xlabel="Timestep", ylabel="Residual (L∞)")
# plot!(corr_rinf, label="Corrector r∞", xlabel="Timestep", ylabel="Residual (L∞)")


# get_pressure!(sim; duration = 3, step = 0.1, remeasure = true, plotbody = true)
# times = [Tp(3), Tp(3+3/5), Tp(3+6/5), Tp(3+9/5), Tp(3+12/5), Tp(3+15/5)]
# sim_frames!(sim; duration = 15, step = 0.1, remeasure=true, plotbody=true, savepath="snapshot", dpi=300)