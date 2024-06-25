using WaterLily
using ParametricBodies
using StaticArrays
using Plots
using CUDA

include("./src/splines.jl")
include("./src/kinematics.jl")

# parameters
function dynamicSpline(;L=2^5, Re=100, U=1, ϵ=0.5, thk=2ϵ+√2, mem=Array)
    # Create the initial shape.    
    cps = shapeForTime(0.0, kinematics_0_baseline, evaluate=false, mirror=true)
    # Position and scale.
    cps = SMatrix{2, 23}(cps[[2, 1], :] .* [-1., 1.] .+ [2,3]) * L

    # needed if control points are moved
    cps_m = MMatrix(cps)
    
    # Create a spline object
    spl = BSplineCurve(cps_m; degree=2)
    
    # Create a nurbs object.
    #weights = SMatrix{1, 23}(zeros(23)')
    #knots = SMatrix{1, 27}(vcat([[0., 0.], range(0, 1, 23), [1., 1.]]...)')
    #spl = NurbsCurve(cps_m, knots, weights)

    # use BDIM-σ distance function, make a body and a Simulation
    # This is for a fillament.
    #dist(p, n) = √(p'*p)-thk/2
    #body = DynamicBody(spl, (0, 1); dist, mem)
    # This is for a closed body.
    body = DynamicBody(spl, (0, 1); mem)
    
    # Set up a sim.
    Simulation((8L, 6L), (U, 0), L; U, ν=U*L/Re, body, T=Float64, mem)
end

# Redefie a function because Marin told me to do so. This should fix issues
# at the ends of the spline.
ParametricBodies.notC¹(l::NurbsLocator, uv) = false

# intialize
sim = dynamicSpline()#mem=CuArray);
t₀, duration, tstep = sim_time(sim), 3., 0.025;

# run
anim = @animate for tᵢ in range(t₀, t₀+duration; step=tstep)

    # update until time tᵢ in the background
    t = sum(sim.flow.Δt[1:end-1])
    
    while t < tᵢ*sim.L/sim.U
        cps = shapeForTime(tᵢ % 1.0, kinematics_0_baseline, evaluate=false, mirror=true)
        new_pnts = SMatrix{2, 23}(cps[[2, 1], :] .* [-1., 1.] .+ [2,3]) * sim.L
        
        ParametricBodies.update!(sim.body, new_pnts, sim.flow.Δt[end])
        measure!(sim, t)
        mom_step!(sim.flow, sim.pois)
        t += sim.flow.Δt[end]
    end

    # Flow plot
    @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L / sim.U
    
    contourf(clamp.(sim.flow.σ, -10, 10)', dpi=300, xlims=(1.5*sim.L, 3.5*sim.L), ylims=(2*sim.L, 4*sim.L),
             color=palette(:RdBu_11), clims=(-10, 10), linewidth=0,
             aspect_ratio=:equal, legend=false, border=:none)
    plot!(sim.body.surf; add_cp=true)
    
    #measure_sdf!(sim.flow.σ, sim.body, WaterLily.time(sim))
    #contour!(sim.flow.σ', levels=[0], color=:magenta, linewidth=2, legend=false)

    # print time step
    println("tU/L=", round(tᵢ, digits=4), ", ΔtU/L=", round(sim.flow.Δt[end]/sim.L*sim.U, digits=3))
end

# save gif
gif(anim, "outputs/plot_04_test_DynamicBody_flow_sdf.gif", fps=10)

