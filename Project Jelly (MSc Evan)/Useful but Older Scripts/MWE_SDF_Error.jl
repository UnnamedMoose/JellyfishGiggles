using StaticArrays, ParametricBodies, Plots
using WaterLily
using BiotSavartBCs
using Interpolations: Flat, LinearInterpolation

T = Tp = Float64
function clamped_uniform_knots(p::Int, Ncp::Int)
    Ninterior = Ncp - p - 1
    head = zeros(Float64, p + 1)
    interior = Ninterior > 0 ? collect(range(0.0, 1.0, length = Ninterior + 2))[2:end-1] : Float64[]
    tail = ones(Float64, p + 1)
    return vcat(head, interior, tail)
end

function make_circle_cps(T::Type, npoints::Int, nsteps::Int; radius=1.0, freq=0.1, amplitude=(1.0, 1.0))
    cps_list = Vector{SMatrix{2, npoints, T}}(undef, nsteps)
    θ = range(0, 2π, length=npoints)
    for k in 0:nsteps-1
        phase = 2π * 3 * (k / (nsteps - 1))    
        shift = (
            amplitude[1] * sin(phase),
            amplitude[2] * cos(phase)
        )
        x = radius .* cos.(θ) .+ shift[1]
        y = radius .* sin.(θ) .+ shift[2]
        cps_list[k+1] = SMatrix{2, npoints, T}(vcat(x', y')...)
    end

    n_cps = length(cps_list[1][1,:]); cps_paths_x = [[] for _ in 1:n_cps]; cps_paths_y = [[] for _ in 1:n_cps]

    for j in 1:length(cps_list)
        for (i, p) in enumerate(cps_list[j][1,:])
            push!(cps_paths_x[i], p)
        end
        for (i, p) in enumerate(cps_list[j][2,:])
            push!(cps_paths_y[i], p)
        end
    end
    return cps_paths_x, cps_paths_y
end

function control_point_functions(sx, sy, t_points)
    N = length(sx)
    interp_funcs = Vector{Function}(undef, N)
    for i in 1:N
        fx = LinearInterpolation(t_points, sx[i], extrapolation_bc=Flat())
        fy = LinearInterpolation(t_points, sy[i], extrapolation_bc=Flat())
        interp_funcs[i] = t -> SA[fx(t), fy(t)]
    end
    return interp_funcs
end

function cps_at_time(interp_funcs, Npoints, t)
    cps_t = SMatrix{2,5,Float64}(hcat([f(t) for f in interp_funcs]...) )
    return cps_t
end

function jelly_sdf(x, t) ## Option, but not differentiable for WaterLily
    D = 2^5; ϵ = 0.75; thk = 2ϵ+√3; deg = 2; n_ctrl = 5
    cps = cps_at_time(pathing, n_ctrl, t)
    weights = ones(Tp, n_ctrl)
    knots = Tp.(clamped_uniform_knots(deg, n_ctrl))
    curve = NurbsCurve(cps .* (D) .+ 2*D, knots, weights)
    body  = DynamicNurbsBody(curve; thk=thk, boundary=true)
    return sdf(body, x, t)
end

jelly_map(x,t) = x

D = 2^5; Re = 302; U = 1; ϵ = 1; thk = 2ϵ+√3; deg = 2  
cycles = 5 
period = 3  
duration = cycles * period  
path_x, path_y    = make_circle_cps(Float64, 5, 100; radius=0.25,freq=0.01, amplitude=(0.25, 0.25))
t_points = range(1,length(path_x[1]), step=1)
pathing = control_point_functions(path_x, path_y, t_points)

plt = Plots.plot()
for i in [1,2,3,4,5]
    pts = [pathing[i](t) for t in t_points]
    xs = getindex.(pts, 1)
    # ys = getindex.(pts, 2)
    Plots.plot!(xs, label="CP $i")
end

cps_start = cps_at_time(pathing, 5, 0)

jelly_shape = AutoBody(jelly_sdf, jelly_map)

sim = BiotSimulation((6D,6D), (0,0), D; U=1, ν=U*D/Re, body=jelly_shape, T=T, mem=Array, ϵ=0.75)

duration=2.5; step=0.1; t₀=round(sim_time(sim))
@time for tᵢ in range(t₀,t₀+duration;step)
    t = sum(sim.flow.Δt[1:end-1])
    while t < tᵢ*sim.L/sim.U
        sim.flow.Δt[end] = WaterLily.CFL(sim.flow; Δt_max=Tp(0.1))
        measure!(sim,t)
        mom_step!(sim.flow,sim.pois)
        t += sim.flow.Δt[end]
    end
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
end