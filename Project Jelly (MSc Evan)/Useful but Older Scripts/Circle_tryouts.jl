using LinearAlgebra, Printf, Statistics, Plots, ParametricBodies, StaticArrays
using WaterLily, CUDA
using GeometryBasics, Optim
using BiotSavartBCs
using Interpolations: Flat, LinearInterpolation
using DelimitedFiles, DataFrames, CSV
using GLMakie
using Dierckx

@info "Running with $(Threads.nthreads()) Julia threads"

Tp = Float64
T = Float64

Tp = Float64
T = Float64

function poly_area(points::Vector{SVector{2,T}}) where T 
    n = length(points)
    sum = zero(T)
    for i in 1:n
        x1, y1 = points[i]
        x2, y2 = points[mod1(i+1, n)]
        sum += x1 * y2 - x2 * y1
    end
    return abs(sum) / 2
end

function get_area(cps)              
    s_vals          = range(0, 1; length=100)            
    curve = BSplineCurve(cps; degree=2)
    points = [curve(s) for s in s_vals]
    area = poly_area(points)
    return area
end

function clamped_uniform_knots(p::Int, Ncp::Int)
    Ninterior = Ncp - p - 1
    head = zeros(Float64, p + 1)
    interior = Ninterior > 0 ? collect(range(0.0, 1.0, length = Ninterior + 2))[2:end-1] : Float64[]
    tail = ones(Float64, p + 1)
    return vcat(head, interior, tail)
end

function gen_p_plots(sim, tᵢ)
    save_dir_p = joinpath(pwd(), "Pressure_check")
    isdir(save_dir_p) || mkpath(save_dir_p)

    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))

    p = Array(sim.flow.p)        
    σ = Array(sim.flow.σ)          
    
    p_masked = copy(p)              
    p_masked[σ .< -ϵ] .= NaN         
    # max_p = maximum(filter(!isnan, p_masked))
    # min_p = minimum(filter(!isnan, p_masked))
    # @show min_p, max_p
    pressure_plot = Plots.heatmap(p_masked', aspect_ratio=1,
    xlims=(1.5sim.L, 4sim.L),
    ylims=(1.5sim.L, 4sim.L),
    c=:balance,          
    clims=(-5, 5),  
    title="Pressure Field")

    Plots.contour!(sim.flow.σ',levels=[-sim.ϵ,0,sim.ϵ])   
    savefig(pressure_plot, joinpath(save_dir_p, "pressure_$(tᵢ).png"))
end

function gen_ω_gif(sim, t, R)
    @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L/sim.U
    @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I]) < 0.001, 0.0, sim.flow.σ[I])
    flood(sim.flow.σ[R] |> Array; clims=(-5, 5))
    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
    Plots.contour!(sim.flow.σ',levels=[-sim.ϵ,0,sim.ϵ], title="$(round(t, digits=4))")
end

function gen_div_gif(sim, t, R)
    @inside sim.flow.σ[I] = WaterLily.div(I,sim.flow.u)
    @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I]) < 0.001, 0.0, sim.flow.σ[I])
    flood(sim.flow.σ[R] |> Array; clims=(-0.5, 0.5))
    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
    Plots.contour!(sim.flow.σ',levels=[-sim.ϵ,0,sim.ϵ], title="$(round(t, digits=4))")
end

function make_circle_timed_sin_motion(T::Type, npoints::Int, t; radius=1.0, freq=0.1, amplitude=(1.0, 1.0))
    # θ = range(0, 2π, length=npoints)

    # shift = (amplitude[1] * sin(2π * freq * t), amplitude[2] * cos(2π * freq * t))
    # x = radius .* cos.(θ) .+ shift[1]
    # y = radius .* sin.(θ) .+ shift[2]
    # cps_list = SMatrix{2, npoints, T}(vcat(x', y')...)
    θ = range(0, 2π, length=npoints)
    x = 0.25 .* cos.(θ) .+ 0.25 * sin(2π * 0.01 * t)
    y = 0.25 .* sin.(θ) .+ 0.25 * cos(2π * 0.01 * t)
    cps_list = SMatrix{2, npoints, T}(vcat(x', y')...)
    return cps_list
end

@inline function TwoDimJellyfish(::Type{T}=Float32; 
    new_cps_list, D=2^7, Re=302, U=1, ϵ=0.5, thk=2ϵ+√3, deg, 
    mem=Array, use_biotsavart=false) where {T<:AbstractFloat}

    cps = new_cps_list[1] .* 1 .* D .+ SA{T}[2D, 2.5D]
    degree = deg
    n_ctrl = size(cps, 2)
    weights = ones(T, n_ctrl)
    knots = T.(clamped_uniform_knots(degree, n_ctrl))
    curve = NurbsCurve(cps, knots, weights)

    body = DynamicNurbsBody(curve; thk=thk, boundary=true)

    ν = U * D / Re

    return use_biotsavart ?
        BiotSimulation((6D, 6D), (0,0), D; U, ν, body, T, mem, ϵ) :
        Simulation((6D, 6D), (0,0), D; U, ν, body, T, mem, ϵ)
end

function simulate_Jelly!(sim, new_cps_list; ThreeD=false,
                         duration=1, period=3, step=0.1, verbose=true,
                         R=inside(sim.flow.p), remeasure=false, plotbody=false, kv...)

    Tp = eltype(sim.flow.p)

    #period = period * sim.L / sim.U
    nphases = length(new_cps_list)

    n_cps = length(new_cps_list[1][1,:])
    cps_paths_x = [[] for _ in 1:n_cps]  # vector of vectors
    cps_paths_y = [[] for _ in 1:n_cps]
    time_points = Float64[]
    forces = Float64[]
    time = Float64[]
    indices = []

    # steps = 3 * duration / nphases
    t₀ = sim_time(sim)

    anim = @animate for tᵢ in range(t₀, t₀ + duration; step=step)
        t = sum(sim.flow.Δt[1:end-1])
        while t < tᵢ * sim.L / sim.U
            # sim.flow.Δt[end] = Tp(sim.L * duration/(nphases))
            # sim.flow.Δt[end] = WaterLily.CFL(sim.flow; Δt_max=Tp(0.05))
            # k = floor(Int, t/(duration*sim.L) * nphases) ; idx0 = mod(k, nphases) + 1 #; idx1 = mod(k + 1, nphases) + 1
            τ = t * sim.U / sim.L
            phase = τ / duration * nphases   # continuous phase in [0, nphases)

            i = mod(floor(Int, phase), nphases) + 1
            j = mod(i, nphases) + 1
            w = phase - floor(phase)

            # θ = range(0, 2π, length=25)
            # x = 0.25 .* cos.(θ) .+ 0.25 * sin(2π * 0.01 * t)
            # y = 0.25 .* sin.(θ) .+ 0.25 * cos(2π * 0.01 * t)
            # cps_interp = SMatrix{2, 25, T}(vcat(x', y')...)
            cps_interp = (1 - w) * new_cps_list[i] + w * new_cps_list[j]   # continuous
            # k = floor(Int, τ / duration * nphases)
            # idx = mod(k, nphases) + 1
            # push!(indices,idx)
            # cps_interp = new_cps_list[idx]             # cps_interp = ( ( (t - (k * period)) / period) * new_cps_list[idx1] + ( ((k + 1) * period - t) / period) * new_cps_list[idx0] )
            if ThreeD
                body_interpolation = cps_interp .* sim.L
            else
                body_interpolation = cps_interp .* sim.L .+ (Tp(2sim.L), Tp(2.5sim.L))
            end
            sim.sim.body = ParametricBodies.update!(sim.sim.body, body_interpolation, sim.flow.Δt[end])
            # measure!(sim,t) 
            # mom_step!(sim.flow,sim.pois) # evolve Flow
            sim_step!(sim, tᵢ; remeasure=true)
            @show sim_time(sim)
            for (i, p) in enumerate(cps_interp[1,:])
                push!(cps_paths_x[i], p)
            end
            # for (i, p) in enumerate(cps_interp[2,:])
            #     push!(cps_paths_y[i], p)
            # end
            push!(time_points, t)
            push!(time, sim_time(sim))

            t += sim.flow.Δt[end]
        end

        force = -WaterLily.total_force(sim)[1] / (0.5*sim.L)
        push!(forces, force)

        gen_p_plots(sim, t)
        gen_ω_gif(sim, t, R, kv...)
        # gen_div_gif(sim, t, R, kv...)

        verbose && println("t=", round(t, digits=4), ", Δt=", round(sim.flow.Δt[end], digits=5))
    end 

    gif(anim,"Swimming_Jelly.gif")

    return (forces=forces, cps_paths_x=cps_paths_x, cps_paths_y=cps_paths_y, time_num=time_points, time_sim=time, indices=indices)
end

function get_forces!(sim, tᵢ, duration, new_cps_list;ThreeD=false)
    # sim.flow.Δt[end] = Tp(0.1)
    nphases = length(new_cps_list)
    phase = tᵢ / duration * nphases

    i = mod(floor(Int, phase), nphases) + 1
    j = mod(i, nphases) + 1
    w = phase - floor(phase)
    cps_interp = (1 - w) * new_cps_list[i] + w * new_cps_list[j]
    if ThreeD
        body_interpolation = cps_interp .* sim.L/2
    else
        body_interpolation = cps_interp .* sim.L .+ (Tp(2sim.L), Tp(2.5sim.L))
    end
    sim.sim.body = ParametricBodies.update!(sim.sim.body, body_interpolation, sim.flow.Δt[end])
    # measure!(sim,t)
    sim_step!(sim, tᵢ; remeasure=true)
    # mom_step!(sim.flow, sim.pois)

    # sim_step!(sim, tᵢ; remeasure=true)
    force = -WaterLily.total_force(sim)[1] 
    @show w
    # push!(forces, force)
    return force
end

ThreeD = false
D = 2^5; Re = 302; U = 1; ϵ = 0.75; thk = 2ϵ+√3; deg = 2  
cycles = 5 # user-defined number of motion cycles
period = 3  # jellyfish motion period is ~1 second
duration = cycles * period  # duration of simulation

function make_circle_cps_sin_motion(T::Type, npoints::Int, nsteps::Int; radius=1.0, freq=0.1, amplitude=(1.0, 1.0))
    cps_list = Vector{SMatrix{2, npoints, T}}(undef, nsteps)

    θ = range(0, 2π, length=npoints)

    for k in 0:nsteps-1
        phase = 2π * 3 * (k / (nsteps - 1))    # sweeps exactly `cycles` times
        shift = (
            amplitude[1] * sin(phase),
            amplitude[2] * cos(phase)
        )
        x = radius .* cos.(θ) .+ shift[1]
        y = radius .* sin.(θ) .+ shift[2]
        cps_list[k+1] = SMatrix{2, npoints, T}(vcat(x', y')...)
    end

    return cps_list
end

new_cps_list    = make_circle_cps_sin_motion(Float64, 25, 100; radius=0.25,freq=0.01, amplitude=(0.25, 0.25))
# new_cps_list, path_x, path_y    = construct_jelly_motion(50,0.001,0.75,12,cycles; ThreeD=ThreeD)
time = range(0,96;step=3)
# new_cps_list = [make_circle_timed_sin_motion(Float64, 25, t; radius=0.25,freq=0.01, amplitude=(0.25, 0.25)) for t in time]
@show new_cps_list
# sim             = ThreeDimJellyfish(; new_cps_list, D, Re, U, ϵ, thk, deg, mem=Array, use_biotsavart=true) 
# sim             = ThreeDimSphere(; new_cps_list, D, Re, U, ϵ, thk, deg, mem=Array, use_biotsavart=true)
sim             = TwoDimJellyfish(; new_cps_list, D, Re, U, ϵ, thk, deg, mem=Array, use_biotsavart=true)
# steps = 3 * duration / length(new_cps_list)
# #duration= 15; t₀=round(sim_time(sim))

# plt = Plots.plot(new_cps_list[1][1,:] .* sim.L, new_cps_list[1][2,:] .* sim.L)
# for i in 2:10
#     Plots.plot!(new_cps_list[i][1,:] .* sim.L, new_cps_list[i][2,:] .* sim.L)
# end
# display(plt)
# t₀ = 0
# time = range(t₀,t₀+duration; step=0.1)
# # results = [get_forces!(sim, tᵢ, duration, new_cps_list; ThreeD=ThreeD) for tᵢ in time]
# res = simulate_Jelly!(sim, new_cps_list; duration=9, period=period, step=0.1, remeasure=true, plotbody=false, ThreeD=ThreeD)
# Plots.plot(res.forces, xlabel="time", ylabel="force", title="force comparison circle", color=:blue, label="idx varying")

radius_c = 0.25 * D
center_c = 2 * D
t0 = 0

times = range(0,10; step=0.1)
# sinusoidal motion
vel(t) = SA[2π * 0.01 * 0.25 * cos(2π * 0.01 * t),
        2π * 0.01 * 0.25 * -sin(2π * 0.01 * t)]

pos(t) = SA[0.25 * 0.1 *sin(2π * 0.01 * t),
    0.25 * 0.1 * cos(2π * 0.01 * t)] 

# collect samples
velo = [vel(t) for t in times]
posi = [pos(t) for t in times]

# extract components
vx = [v[1] for v in velo]
vy = [v[2] for v in velo]
px = [p[1] for p in posi]
py = [p[2] for p in posi]

# plot both components
plt = Plots.plot(times, vx, label="vₓ", xlabel="time", ylabel="value")
Plots.plot!(plt, times, vy, label="v_y")
Plots.plot!(plt, times, px, label="pₓ", linestyle=:dash)
Plots.plot!(plt, times, py, label="p_y", linestyle=:dash)
display(plt)

# mapping function that subtracts the motion from coordinates
map(x, t) = x - pos(t) * D

# static circle geometry
circle = AutoBody(
    (x,t) -> sqrt(sum(abs2, x .- center_c)) - radius_c,
    map
)

sim2 = BiotSimulation((6D,6D), (1,0), D; U=1, ν=U*radius_c/Re, body=circle, T=T, mem=Array, ϵ=0.75)
forces = []
time = []
time_sim = []
forces_out = []
duration=10; step=0.1; t₀=round(sim_time(sim2))
@time @gif for tᵢ in range(t₀,t₀+duration;step)

    # update until time tᵢ in the background
    t = sum(sim2.flow.Δt[1:end-1])
    while t < tᵢ*sim.L/sim.U

        # measure body
        measure!(sim2,t)

        # update flow
        mom_step!(sim2.flow,sim2.pois)

        # pressure force
        force = -WaterLily.total_force(sim2)
        push!(forces, force[1])
        push!(time, t)
        push!(time_sim, sim_time(sim2))
        # compute motion and acceleration 1DOF
        Δt = sim2.flow.Δt[end]
        # accel = (force[2]- k*p0 + mₐ*a0)/(m + mₐ)
        # p0 += Δt*(v0+Δt*accel/2.)
        # v0 += Δt*accel
        # a0 = accel

        # update time, sets the pos/v0 correctly
        t += Δt; t0 = t
    end
    force_out = -WaterLily.pressure_force(sim2)
    push!(forces_out, force_out[1])
    # plot 
    R = inside(sim.flow.p)
    gen_ω_gif(sim2, t, R)
    # print time step
    println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim2.flow.Δt[end],digits=3))
end