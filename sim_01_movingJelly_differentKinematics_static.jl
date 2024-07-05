using WaterLily
using ParametricBodies
using BiotSavartBCs
using StaticArrays
using Plots; ENV["GKSwstype"]="nul"
using DelimitedFiles
using ReadVTK, WriteVTK
using LoggingExtras

include("./src/splines.jl")
include("./src/kinematics.jl")
include("./src/tools.jl")

caseId = "baseline"
kinematics_arr = kinematics_0_baseline

#caseId = "noFlick"
#kinematics_arr = kinematics_1_noFlick

#caseId = "largeFlick_pTol_1e-2"
#kinematics_arr = kinematics_2_largeFlick

ReynoldsNumber = 500.
maxTipVel = 3.5  # From kinematics for the "Large flick" case
L = 32
Uinf = 1.0
period = maxTipVel*L/Uinf

tstep = 0.005*period
duration = 2.0*period
centre = [1.5, 1.5]*L

#function map(x, t)
#    SA[1. 0.; 0. 1]*(x-centre)
#end

function dynamicSpline(;Re=500, U=1, mem=Array)
    # Create the initial shape.    
    cps = shapeForTime(0.0, kinematics_arr, evaluate=false, mirror=true)
    # Position and scale.
    cps = SMatrix{2, 23}(cps[[2, 1], :] .* [-1., 1.]) * L .+ centre

    # needed if control points are moved
    cps_m = MMatrix(cps)
    
    # Create a spline object
    spl = BSplineCurve(cps_m; degree=3)

    # use BDIM-σ distance function, make a body and a Simulation
    body = DynamicBody(spl, (0, 1); mem)
    
    # Set up a sim.
    # TODO
    Simulation((4L, 3L), (0, 0), L; U, ν=U*L/Re, body, T=Float64, mem)
end

# Redefie a function because Marin told me to do so. This should fix issues
# at the ends of the spline.
ParametricBodies.notC¹(l::NurbsLocator, uv) = false

# Set convergence tolerances.
WaterLily.solver!(ml::MultiLevelPoisson) = WaterLily.solver!(ml; tol=1e-6, itmx=1024)

# Intialize
sim = dynamicSpline(Re=ReynoldsNumber, U=Uinf)
t₀ = sim_time(sim)

# Set Biot-Savart BC (or not)
using_biot_savart = true
ω_ml = using_biot_savart ? MLArray(sim.flow.σ) : nothing

# Allows logging the pressure solver results.
WaterLily.logger("log_psolver")

# Keeps time series data
global timeHistory = []

# make a writer with some attributes, need to output to CPU array to save file (|> Array)
velocity(sim::Simulation) = sim.flow.u |> Array;
pressure(sim::Simulation) = sim.flow.p |> Array;
_body(sim::Simulation) = (measure_sdf!(sim.flow.σ, sim.body, WaterLily.time(sim)); sim.flow.σ |> Array;)

custom_attrib = Dict(
    "Velocity" => velocity,
    "Pressure" => pressure,
    "Body" => _body
)

wr = vtkWriter("flowDataFile_" * caseId; attrib=custom_attrib)

# ===
#for tᵢ in range(t₀, t₀+duration; step=tstep)
anim = @animate for tᵢ in range(t₀, t₀+duration; step=tstep)
# ===

    # update until time tᵢ in the background
    t = sum(sim.flow.Δt[1:end-1])
    
    while t < tᵢ
        cps = shapeForTime(t/period % 1.0, kinematics_arr, evaluate=false, mirror=true)
        new_pnts = SMatrix{2, 23}(cps[[2, 1], :] .* [-1., 1.]) * sim.L .+ centre
        
        ParametricBodies.update!(sim.body, new_pnts, sim.flow.Δt[end])
        
        measure!(sim, t)
        using_biot_savart ? biot_mom_step!(sim.flow, sim.pois, ω_ml) : mom_step!(sim.flow, sim.pois)
        
        t += sim.flow.Δt[end]
    end
    
    # Grab forces and store them.
    # fTot = -WaterLily.∮nds(sim.flow.p, sim.flow.f, sim.body, t)
    fTot = -ParametricBodies.∮nds(sim.flow.p, sim.body, t)
    push!(timeHistory, [sim.flow.Δt[end], t, fTot...])
    
    # Save to vtk.
    write!(wr, sim)

# ===

    # Flow plot
    #@inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L / sim.U; lims = 10
    #@inside sim.flow.σ[I] = sim.flow.p[I]; lims = 1
    @inside sim.flow.σ[I] = sim.flow.V[I]; lims = 0.2
    
    contourf(clamp.(sim.flow.σ, -lims, lims)', dpi=300, xlims=(centre[1]-0.25*sim.L, centre[1]+1.25*sim.L),
            ylims=(centre[2]-0.75*sim.L, centre[2]+0.75*sim.L),
            color=palette(:RdBu_11), clims=(-lims, lims), linewidth=0,
            aspect_ratio=:equal, legend=false, border=:none)
    
    # Body plot.
    measure_sdf!(sim.flow.σ, sim.body, WaterLily.time(sim))
    contour!(sim.flow.σ', levels=[0], color=:magenta, linewidth=2, legend=false, show=true)

# ===

    # print time step
    println("t/T=", round(tᵢ/period, digits=4), ", Δt/T=", round(sim.flow.Δt[end]/period, digits=6))
    
    # Flush them outputs.
    flush(stdout)
end

# Convert to a column matrix for plotting and saving.
timeHistory = hcat(timeHistory...)'
writedlm("outputs/timeHistory_" * caseId * "_Re_$ReynoldsNumber.csv", timeHistory, ',')

# ===

# save gif
gif(anim, "outputs/plot_04_flow_sdf_" * caseId * "_Re_$ReynoldsNumber.gif", fps=10)

# Plot the force
plot(timeHistory[:, 2], timeHistory[:, 3], legend=false, xlabel="Time", ylabel="Propulsive force")
savefig("outputs/plot_04_force_" * caseId * "_Re_$ReynoldsNumber.png")

# ===

# Clean up the VTK writer.
close(wr)

# ===
# Read the pressure log.
# include("./src/tools.jl"); using Plots; plot_logger("log_psolver.log")

