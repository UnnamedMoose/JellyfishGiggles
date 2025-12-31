"""
Include the background packages where functions are formed.
"""

include("Background_functions.jl")
include("Visualisation.jl")
include("Validation_arrays.jl")
include("Prolate_cps.jl")
# include("Oblate_cps.jl")      # This way, the control point set for oblate jellyfish can be used.

""" --- Simulation Parameters --- 
In Marin his paper, he says that grid resolution controls how well thin structures are captured. It always has to cover at least some cells. It is important to keep track of.
Fineness Ratio = D/h
Good to discuss possible active control parameters. Shape profile could be easily researched.
"""

## ALL INPUT PARAMS:
Dmax            = 1.25              # Maximum jellyfish diameter (from Sahin 2009)
Ncps            = 50                # Number of control points to define jellyfish body  
n_cycles        = 35                # Number of cycles to be simulated.
n_up            = 10                # General number of upsamples
γ               = 1.0               # Expansion to contraction phase ratio

## NUMERICAL SETTINGS
D               = 2^5               # Grid size of maximum Jellyfish diameter       D = 2ᵖ for p = 4,5,6,7 (8?)
ϵ               = 1                 # BDIM interface thickness, this means the thickness is 1 grid cell (which is standard in examples)
deg             = 1                 # Polynomial degree to describe jellyfish boundary
Domain          = 4D                # Domain size
Uff             = 0                 # 'far-field' velocity
U               = 1                 # Reference velocity

## REFERENCE VALUES
Re              = 302               # Reynolds Number (From Sahin 2009) (UD/ν) based on avg medusa velocity of 2.42 cm/s
St              = 0.52              # Strouhal number (From Sahin 2009) (D/(UT)) based on avg medusa velocity of 2.42 cm/s
ν               = U * D / Re        # Derivation of the numerical viscosity
CFD_t_scale     = Dcfd / Ucfd       # Time scale derived from the characteristic length and velocity in (Sahin 2009)
period_CFD      = 1 / CFD_t_scale   # Non-dimensional motion period 
WL_t_scale      = D/U               # WaterLily time scale based on the characteristic length and velocity I defined
period_WL       = period_CFD * WL_t_scale       # Period in WaterLily time units

""" --- Control Point Generation ---
Generation Process:
1. A set of control points, digitized from Sahin 2009, representing the bell kinematics of the (half) jellyfish, is used to define the motion of the body.
2. This set of control points is split into a contraction and expansion phase, with the starting cps set of the next phase added at the end of the current phase to ensure continuity.
3. The number of control points to define each time step is extended to a number of Ncps (first input of constructor).
4. Next, the number of frames to define a full cycle is expanded, using upsampling.
    This spline method defines 2 '1D splines' for each control point, to describe its motion in x- and y-direction as a spline.
    From this spline, it derives a new and extended pathing for each contorl point coordinate. The spline should ensure continuity of the pathing, as it is a parametrisation.
    The number of frames to define a full cycle is contorlled by the upsample variable up. # upsamples is added in between each original frame.
5. The new set of frames with control points is then mirrorred and the control point order reversed to acquire the right definition of the body for WaterLily.jl
6. It outputs the paths of each coordinate (for x and y), the contraction and expansion control point sets and the full cps list.

In terms of shape and sdf, the Ncps (seems good at 50), deg 1.

Assign each cps_set to its specific phase:
"""

phase_contr         = [cps_0, cps_1, cps_2, cps_3, cps_4] ./ Dmax               # 0, T/10, 2T/10, 3T/10, 4T/10 -> Required for correct upsampling.
phase_exp           = [cps_4, cps_5, cps_6, cps_7, cps_8, cps_9] ./ Dmax        # 4T/10, 5T/10, 6T/10, 7T/10, 8T/10, 9T/10
pathing, period_fr  = generate_jelly_motion(phase_contr, phase_exp, Ncps, n_cycles, n_up, γ)

""" --- Control Point Generation Debugging ---
Different plotting routines to check if all functions are well-behaved and the results create continuous CPS motion.
Particularly for control point 33, which is the outer point of the jellyfish flap is good for checks.
Function to display a plot of the actual jelly shape at a specific frame. 
Check the kinematic profile compared to (Sahin 2009)
"""

# Check the pathing of a specific control point:
plt = Plots.plot(pathing[20](collect(0:1:500))[1], xlabel="frame number", ylabel="x-coordinate", title="x-pathing of CP33"); hline!(plt, [pathing[cp_check](0)[1]]); vline!(plt, [1, 2, 3, 4, 5] .* t_fr_scaling); display(plt)                      

# Generate kinematic plots on top of the reference Figures from (Sahin 2009). Switch to expansion with contr=false.
generate_kin_checks(pathing, period_fr, Ncps; contr=false)

# Check the signed distance field
signed_distance_field(deg, D, Re, U, Domain, Uff, ϵ, pathing, period_CFD, period_fr)

# Check the grid size on the flap (proves D=2^7 is good)
gridsize_on_flap(pathing, Ncps, D, Domain)

""" --- Simulation Setup 2 ---
In this part the actual simulation can be run. 
1. Input for this is all defined above using the simulation parameters. The starting position of the jellyfish is defined in the above section from cps_start.
2. Define the starting location for the jellyfish.
3. Setup the simulation 'environment' for the jellyfish. Requires input from the simulation parameters above.
4. Viscosity is calculated from velocity and length scale and the Reynolds number.
5. The body is constructed using the DynamicNurbsBody from the ParametricBodies package.
6. The knots array is automatically generated from the polynomial degree and number of control points.
7. Weights are all put to 1 for all control points.
8. Can be setup with either the BiotSavartBCs or a general WaterLily simulation.
9. The body is updated each timestep using the ParametricBodies update functionality.
10. The sim / flow is updated with measure! and biot_mom_step!, which is an implementation of a moving grid.
11. Pressure solver statistics can be put on to check the performance of the solver.

Currently this is able to compute force, acceleration, velocity, displacement, pressure plots and vorticity plots/gif.
All output is generated as the simulation runs and written into a .csv-file.
Plotting routines to check the results can be used after the simulation is run. 

Added Mass from MCHenry 2003: Hemiellipsoidal approach. An estimate, but skipping the effect of the flap.
A = α * ρ * Vₛ * acc
α = (2h/d)^(1.4)        Added mass coefficient, h and d are the bell shape parameters.
Vₛ = π * d^2 * hₛ / 6    hₛ is the cavity height, d is cavity diameter, both supposedly a function of time.
For Sarsia sp. the max diameter is 1.25 cm during expansion. 1.15 cm during contraction.
Height is 1.20 cm during the expansion phase. 1.40 cm during contraction.
Cavity volume is very variable, average at about 0.80 cm.

--- Making the Jellyfish Move Forward ---
Solving Newton's F = ma for the acceleration, using the hydrodynamic forces resulting from the solver.
Integrate the acceleration to a velocity and integrate that for its position, 
although for the full problem, only velocity and acceleration are required, as those are the input for the moving grid.
It is helpful to check the displacement of the jellyfish as well.
"""

# Add VTKWriter to evaluate results in ParaView.

WaterLily.logger("Data/Simulation_Data/test_psolver")
run_jelly_simulation(period_WL, period_fr, D, Re, U, ϵ, pathing, Domain, Uff, Ncps)
plot_logger("Data/Simulation_Data/test_psolver")
savefig("Figures/psolver.png")

""" --- Simulation results bookkeeping --- 
All data written to a CSV can be retracted again here and used for plots and stuff.

If time suffices, translate the Python statistical convergence tool to Julia and implement it here.
Move from a periodic to a phase average plotting routine.
"""

# Write into a DataFrame.
df = CSV.read("Data/Simulation_Data/results.csv", DataFrame)
force = df[:,1]; force_am = df[:,2]; force_in = df[:,3]; force_dr = df[:,4]; nd_time = df[:,5]; displacement = df[:,6]; velocity = df[:,7]; acceleration = df[:,8]; enstrophy = df[:,9]

# Choose a signal from the read results file and generate a plot.
signal = force
signal_plot(nd_time, signal, n_cycles; skip_period=true)

# Create a gif from a folder filled with frames of either velocity, pressure or vorticity -> Change to VTK
create_gif_from_folder("Figures/Velocity_check/", "Figures/velocity_output.gif", delay=0.05)
create_gif_from_folder("Figures/Pressure_check/", "Figures/pressure_output.gif", delay=0.05)
create_gif_from_folder("Figures/Vorticity_check/", "Figures/vorticity_output.gif", delay=0.05)

""" --- Validation of the 2D model ---
Use the kinematics in Jellyfish_Kinematics.xlsx to validate the results of the 2D WaterLily model.
In the file Validation Arrays, the data is stored in arrays. Periodic signals are stored as 1 period. The experimental data as full sets.

The following validation data is available:
Periodic: CFD_Cd (total drag coef), CFD_Cdf (total friction drag coef), CFD_Cdp (total thrust coef), CFD_Cp (power coef), CFD_vel_per (periodic part of the velocity)
Non-periodic: exp_acc (acceleration), exp_vel (velocity), exp_displ (position), exp_re (Reynolds), CFD_Velocity (full velocity set)

Choose a signal with time = signal[1,:]; signal = signal[2,:]
Use periodic data with make_periodic_from(signal, scale) (creates a cubic interpolation fit for extrapolation)
Use non-periodic data with make_window_from(signal, scale) (creates a linear interpolation fit)
"""

t₀ = 0; t_end = 5; dt_per = 0.001; dt_win = 0.01

signal, T = make_periodic_from(CFD_vel_per)

tsp = t₀:dt_per:t_end*T; ysp = sample_signal(signal, tsp)

window, Twin = make_window_from(exp_vel)

ts = t₀:dt_win:t₀+Twin; ys = sample_signal(window(t₀), ts)

Plots.plot(ts, -ys / mean(ys), xlims=(t₀, t_end), xlabel="tU/D", ylabel="u/U", title="Jellyfish Forward Velocity", label="Experimental Velocity", legend=:topright)
Plots.plot(tsp, ysp,  xlims=(t₀, t_end), xlabel="tU/D", ylabel="u/U", title="Jellyfish Forward Velocity", label="CFD velocity (Sahin 2009)")
Plots.plot!(nd_time , velocity * Ucfd, label="WaterLily Velocity")

""" --- A 3D Jellyfish ---
Use a mapping function to revolve the 2D jellyfish around its axis of symmetry.
"""

function run_3D_jelly_simulation(period, period_fr, D, Re, U, ϵ, pathing, Domain, Uff, Ncps)
    xloc = 0.5D; yloc = 0; Domain_x = 3Domain #Int(Domain / 2)
    rev_map(x,t)    = SA[(x[1]), hypot(x[2], x[3]) ] 
    cps         =   cps_at_time(pathing, 2*Ncps+5, 0;) .+ SA{T}[0.5D; 0]
    weights     =   ones(T, size(cps, 2)); knots       =   Float64.(knots_vector(deg, size(cps, 2))); curve       =   NurbsCurve(cps, knots, weights )
    body        =   ParametricBody(curve; map=rev_map, ndims=3)
    return BiotSimulation((Domain_x, Domain, Domain), (Uff,Uff,Uff), D; U, ν, body, T, mem=Array, ϵ)
end

sim                 = run_3D_jelly_simulation(period, period_fr, D, Re, U, ϵ, pathing, Domain, Uff, Ncps)

Makie.inline!(false)

begin
    t = 0.05
    xloc = 0.5D; yloc = 0; Domain_x = 3Domain #Int(Domain / 2)
    t₀ = round(sim_time(sim)); step = 0.1; t0 = 0; a0 = 0; v0 = 0; p0 = 0

    cps             = cps_at_time(pathing, 2*Ncps+5, t*(period_fr/(period))) .* D .+ SA{T}[xloc, yloc]
    d = maximum(cps[2,:]) - yloc; h = maximum(cps[1,:]) - xloc; α = (2*h / d)^1.4

    sim.sim.body    = ParametricBodies.update!(sim.sim.body, cps, sim.flow.Δt[end])
    measure!(sim)
    biot_mom_step!(sim.flow,sim.pois,sim.ω,sim.x₀,sim.tar,sim.ftar;
                    fmm=sim.fmm,udf=fall!,acceleration=SA[-a0,0.f0,0.f0],U=SA[-v0,0.f0,0.f0])

    force           =   -WaterLily.pressure_force(sim)[1]
    @show force
    Δt              =   sim.flow.Δt[end]
    force_dr        =   24 / (Re^(0.7)) * 0.5 * get_area(cps) * v0
    accel           =   (force + α * get_area(cps) * a0) / (get_area(cps) * (1 + α))
    force_in        =   get_area(cps) * accel
    force_am        =   α * get_area(cps) * (accel - a0)
    
    p0              +=  Δt * (v0 + Δt * accel / 2.)
    v0              +=  Δt * accel
    a0              =   accel

    # Create CPU buffer arrays for geometry flow viz 
    a               = sim.flow.σ
    d               = similar(a,size(inside(a))) |> Array; # one quadrant
    md              = similar(d, (1,2,2).*size(d))

    # Set up geometry viz
    geom            = geom!(md,d,sim) |> Observable;
    ω               = ω!(md, d, sim) |> Observable

    fig             = GLMakie.Figure()
    ax              = GLMakie.Axis3(fig[1, 1], aspect = :data)

    GLMakie.volume!(ax, ω;algorithm=:mip,transparency=true,alpha=0.45,colormap=:algae,colorrange=(1,10))
    GLMakie.mesh!(ax, geom, alpha=0.6, color=:red)

    fig
end

nframes = 100
@info "Generating $nframes frames..."


isdir("frames") || mkpath("frames")
for frame in 51:nframes
    @show frame

    cps = cps_at_time(pathing, 105, frame) .* D .+ SA_F32[0.5D; 0]
    sim.sim.body = ParametricBodies.update!(sim.sim.body, cps, sim.flow.Δt[end])

    sim_step!(sim,sim_time(sim)+0.05; remeasure = true);    

    geom[]  = geom!(md, d, sim)
    ω[] = ω!(md, d, sim)

    # ---- Save frame ----
    fn = @sprintf("frames/frame_%04d.png", frame)
    save(fn, fig)
end

create_gif_from_folder("frames/", "frames/output.gif", delay=0.05)

"""
3D Moving Jellyfish with a moving grid.

The results we are aiming for that say something regarding the swimming and propulsive performances of the jellyfish under certain settings.
Cost of Transport = Energy / (mass * Velocity_avg)
η_froude = (Thrust * Velocity) / Power
Strouhal = (Frequency * Diameter) / Velocity = Diameter / (Velocity * Thrust)
Power coefficient = 2 * Power / (ρ * Velocity³ * π * (Diameter/2)²)
Focus on:                           WaterLily Version
- Force                             Cf = 2 * WL.total_force / sim.L * sim.U²
- Velocity                          Cu = velocity / sim.U
- Mass/Volume/Diameter              Cvol = volume / sim.L³
- Time/Frequency                    Ct = t * sim.L / sim.U
"""