using WaterLily
using ParametricBodies
using StaticArrays
using Plots
using CUDA

include("./src/splines.jl")

# Spline control points for regressing the motion of each jellyfish segment.
# We need an implementation compatible with GPUs. This necessitates the
# use of static arrays.

Nsegments = 6

cps_thetas = SMatrix{Nsegments+1, 7}(vcat([
# t/T values
[0, 0.025, 0.15, 0.25, 0.35, 0.45, 1.0]',
# CP locations
hcat([
    [0., 0.,        0., 0., 0.,       0., 0.],
    [0., 0.,        0., 0., 0.,       0., 0.],
    [0.39, 0.39,        0., 0.38, 0.39,       0.39, 0.39],
    [0.4, 0.4,        0.85, 1.2, 0.41,       0.4, 0.4],
    [0.95, 0.95,        1.95, 1.65, 0.95,       0.95, 0.95],
    [2.2, 2.2,        -1., 2.19, 2.2,       2.2, 2.2],
]...)'
]...));

cps_lengths = SMatrix{Nsegments+1, 7}(vcat([
[0, 0.025, 0.15, 0.25, 0.35, 0.45, 1.0]',
hcat([
    [0.02, 0.02,        0.01, 0.03, 0.022,       0.02, 0.02],
    [0.04, 0.04,        0.02, 0.06, 0.04,       0.04, 0.04],
    [0.12, 0.12,        0.1, 0.155, 0.11,       0.12, 0.12],
    [0.194, 0.194,        0.275, 0.194, 0.194,       0.194, 0.194],
    [0.228, 0.228,        0.16, 0.225, 0.228,       0.228, 0.228],
    [0.250, 0.250,        0.135, 0.245, 0.250,       0.250, 0.250],
]...)'
]...));

cps_halfThicknesses = SMatrix{Nsegments+1, 7}(vcat([
[0, 0.025, 0.15, 0.25, 0.35, 0.45, 1.0]',
hcat([
    [0.097, 0.097,        0.1005, 0.0875, 0.097,       0.097, 0.097],
    [0.097, 0.097,        0.1005, 0.0875, 0.097,       0.097, 0.097],
    [0.083, 0.083,        0.11, 0.075, 0.083,       0.083, 0.083],
    [0.063, 0.063,        0.075, 0.04, 0.063,       0.063, 0.063],
    [0.0335, 0.0335,        0.045, 0.0345, 0.034,       0.0335, 0.0335],
    [0.011, 0.011,        0.011, 0.011, 0.011,       0.011, 0.011],
]...)'
]...));

# parameters
function dynamicSpline(;L=2^6, Re=100, U=1, ϵ=0.5, thk=2ϵ+√2, mem=Array)
    # Create the initial shape.    
    cps = shapeForTime(0.0, evaluate=false, mirror=true)
    # Position and scale.
    cps = SMatrix{2, 23}(cps[[2, 1], :] .* [-1., 1.] .+ [2,3]) * L

    # needed if control points are moved
    cps_m = MMatrix(cps)
    
    # Create a spline object
    spl = BSplineCurve(cps_m; degree=2)

    # use BDIM-σ distance function, make a body and a Simulation
    dist(p, n) = √(p'*p)-thk/2
    body = DynamicBody(spl, (0, 1); dist, mem)
    Simulation((8L, 6L), (U, 0), L; U, ν=U*L/Re, body, T=Float64, mem)
end

# intialize
sim = dynamicSpline()#mem=CuArray);
t₀, duration, tstep = sim_time(sim), 0.1, 0.0125;

# run
anim = @animate for tᵢ in range(t₀, t₀+duration; step=tstep)

    # update until time tᵢ in the background
    t = sum(sim.flow.Δt[1:end-1])
    
    while t < tᵢ*sim.L/sim.U
        cps = shapeForTime(tᵢ % 1.0, evaluate=false, mirror=true)
        new_pnts = SMatrix{2, 23}(cps[[2, 1], :] .* [-1., 1.] .+ [2,3]) * sim.L
        
        ParametricBodies.update!(sim.body, new_pnts, sim.flow.Δt[end])
        measure!(sim, t)
        mom_step!(sim.flow, sim.pois)
        t += sim.flow.Δt[end]
    end

    # Flow plot
         
    @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L / sim.U
    
    contourf(clamp.(sim.flow.σ, -10, 10)', dpi=300, xlims=(1*sim.L, 4*sim.L), ylims=(2*sim.L, 4*sim.L),
             color=palette(:RdBu_11), clims=(-10, 10), linewidth=0,
             aspect_ratio=:equal, legend=false, border=:none)
    plot!(sim.body.surf; add_cp=true)
    
    measure_sdf!(sim.flow.σ, sim.body, WaterLily.time(sim))
    contour!(sim.flow.σ, levels=[-0.5, 0, 0.5], color=:black, linewidth=0.5, legend=false)

    # print time step
    println("tU/L=", round(tᵢ, digits=4), ", ΔtU/L=", round(sim.flow.Δt[end]/sim.L*sim.U, digits=3))
end

# save gif
gif(anim, "outputs/plot_04_test_DynamicBody_flow_sdf.gif", fps=10)

