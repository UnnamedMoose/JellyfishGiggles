include("New_Simu_Trial.jl")
include("Jellyfish_3D.jl")
ThreeD = false
D = 2^5; Re = 302; U = 1; ϵ = 0.5; thk = 2ϵ+√3; deg = 2  
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

# new_cps_list    = make_circle_cps_sin_motion(Float64, 25, 100; radius=0.25,freq=0.01, amplitude=(0.25, 0.25))
path_x, path_y, cps_contr, cps_exp, cps_list_new    = construct_jelly_motion(50,0.001,0.75,5,1; ThreeD=ThreeD)
function blend_cycles(v::Vector{Any}, n_cycles::Int; overlap_ratio=0.1) where {T<:Real}
    n = length(v)
    overlap = round(Int, n * overlap_ratio)
    result = copy(v)

    for _ in 2:n_cycles
        a = v[end-overlap+1:end]
        b = v[1:overlap]
        blend = (1 .- range(0, 1, length=overlap)) .* a .+ range(0, 1, length=overlap) .* b
        result = vcat(result[1:end-overlap], blend, v[overlap+1:end])
    end
    return result
end
path_x = [blend_cycles(p, 5) for p in path_x]
path_y = [blend_cycles(p, 5) for p in path_y]

function exp_smooth(x::Vector{Any}, α::T) where {T<:Real}
    s₀ = similar(x)     # First do a forward pass
    s₀[1] = x[1]
    for t in 2:length(x)
        s₀[t] = α * x[t] + (1 - α) * s₀[t-1]
    end
    s₁ = similar(x)     # Then do a backward pass
    s₁[end] = s₀[end]
    for t in (length(x)-1):-1:1
        s₁[t] = α * s₀[t] + (1 - α) * s₁[t+1]
    end
    return s₁
end

len = length(path_x)
path_x_smooth = [exp_smooth(path_x[i], 0.250) for i in 1:len]
path_y_smooth = [exp_smooth(path_y[i], 0.250) for i in 1:len]

function control_point_functions(sx, sy, t_points)
    N = length(sx)
    vel_funcs = Vector{Function}(undef, N)
    interp_funcs = Vector{Function}(undef, N)
    for i in 1:N
        fx = LinearInterpolation(t_points, sx[i], extrapolation_bc=Flat())
        fy = LinearInterpolation(t_points, sy[i], extrapolation_bc=Flat())
        vx = [diff(path_x[i]) ./ diff(t_points) for i in 1:N]
        vy = [diff(path_y[i]) ./ diff(t_points) for i in 1:N]

        ax = [diff(vx[i]) ./ diff(t_points[2:end]) for i in 1:N]
        ay = [diff(vy[i]) ./ diff(t_points[2:end]) for i in 1:N]
        
        velx = LinearInterpolation(t_points[2:end], vx[i], extrapolation_bc=Flat())
        vely = LinearInterpolation(t_points[2:end], vy[i], extrapolation_bc=Flat())

        accx = LinearInterpolation(t_points[3:end], ax[i], extrapolation_bc=Flat())
        accy = LinearInterpolation(t_points[3:end], ay[i], extrapolation_bc=Flat())

        vel_funcs[i] = t -> SA[velx(t), vely(t)]
        interp_funcs[i] = t -> SA[fx(t), fy(t)]
    end
    return vel_funcs, interp_funcs
end

function cps_at_time(interp_funcs, Npoints, t)
    cps_t = SMatrix{2,Npoints,Float64}(hcat([f(t) for f in interp_funcs]...) )
    return cps_t
end

t_points = range(1,length(path_x_smooth[25]), step=1)
velocity, pathing = control_point_functions(path_x_smooth, path_y_smooth, t_points)

# vel_25 = [pathing[25](t) for t in t_points]
# display(Plots.plot(t_points[2:end], vel_25))

# plt = Plots.plot()
# for i in [1, 10, 50]
#     pts = [pathing[i](t) for t in t_points]
#     xs = getindex.(pts, 1)
#     # ys = getindex.(pts, 2)
#     Plots.plot!(xs, label="CP $i")
# end
# Plots.plot!(xlabel="x", ylabel="y", legend=:false)
# display(plt)

cps_start = cps_at_time(pathing, 105, 0) # defined from t = 0 to t = 545, which are actually frames.

# cps25_traj = cps_at_time(pathing, 105, collect(0:1:545))[25] * D
cps33_traj = pathing[33](collect(0:1:500))[1]
# cps25_traj_vel = pathing[25](0)[1] .+ accumulate(+,[velocity[25](i)[1] for i in 0:1:500])


# cps25_traj_vel = cps_start[25] + cps_at_time(velocity, 105, collect(0:1:545))[25] * 0.1 * D
plt = Plots.plot(cps33_traj, xlabel="time",ylabel="x-coordinate CP33", title="CP33 Displacement")
# Plots.plot!(cps25_traj_vel)
display(plt)
# plt = Plots.plot(cps_start[1,:], cps_start[2,:])
# for t in 41:55
#     cps_n = cps_at_time(pathing, 105, t)
#     Plots.plot!(cps_n[1,:], cps_n[2,:])
# end
# display(plt)

# time = range(0,96;step=3)
# # new_cps_list = [make_circle_timed_sin_motion(Float64, 25, t; radius=0.25,freq=0.01, amplitude=(0.25, 0.25)) for t in time]
# @show new_cps_list
# # sim             = ThreeDimJellyfish(; new_cps_list, D, Re, U, ϵ, thk, deg, mem=Array, use_biotsavart=true) 
# # sim             = ThreeDimSphere(; new_cps_list, D, Re, U, ϵ, thk, deg, mem=Array, use_biotsavart=true)

# function jelly_sdf(x, t) ## Option, but not differentiable for WaterLily
#     D = 2^5; Re = 302; U = 1; ϵ = 0.75; thk = 2ϵ+√3; deg = 2; n_ctrl = 105
#     cps = cps_at_time(pathing, n_ctrl, t)
#     weights = ones(Tp, n_ctrl)
#     knots = Tp.(clamped_uniform_knots(deg, n_ctrl))
#     curve = NurbsCurve(cps .* (D) .+ 2*D, knots, weights)
#     body  = DynamicNurbsBody(curve; thk=thk, boundary=true)
#     return sdf(body, x, t)
# end
# control points defining bell shape

function jelly_sdf(x, t)
    D = 2^5; ϵ = 0.75; thk = 2ϵ+√3; deg = 2; n_ctrl = 105
    cps = cps_at_time(pathing, n_ctrl, t)
    weights = ones(Tp, n_ctrl)
    knots = Tp.(clamped_uniform_knots(deg, n_ctrl))
    curve = NurbsCurve(cps .* (D), knots, weights)
    body  = DynamicNurbsBody(curve; thk=thk, boundary=true)
    # display(Plots.plot(body))
    # dmin = minimum(norm(x .- cp) for cp in cps_list_new[1])
    return sdf(body,x,t)   # 0.05 = approximate body thickness
end

jelly_map(x,t) = AbstractMatrix{2,105,Float64,210}(cps_updates)
jelly_shape = AutoBody(jelly_sdf, jelly_map)

xs = range(-6.25, 65, length=200)
ys = range(-20, 20, length=200)
Z = [jelly_sdf([x,y], 1) for y in ys, x in xs]

plt = Plots.plot(cps_at_time(pathing,105,0)[1,:]*D, cps_at_time(pathing,105,0)[2,:]*D; aspect_ratio=:equal, legend=:topright,
           title="Jellyfish geometry evolution", xlabel="x", ylabel="y")
Plots.contour!(xs, ys, Z)
display(plt)

# sim = BiotSimulation((6D,6D), (0,0), D; U=1, ν=U*D/Re, body=jelly_shape, T=T, mem=Array, ϵ=0.75)
function run_jelly_simulation(cps_start, D, Re, U, ϵ, thk, deg, pathing)
    sim = TwoDimJellyfish(; new_cps_list=cps_start, D, Re, U, ϵ, thk, deg, mem=Array, use_biotsavart=true)
    forces = []; forces_out = []; time = []; time_sim = []; timesteps = []; displacement = []; velocity = []; acceleration = []
    n_cps = length(cps_start)
    cps_paths_x = [[] for _ in 1:n_cps]
    cps = cps_start .* D .+ SA{T}[2D, 2.5D]
    duration = 10
    step = 0.1
    t₀ = sim_time(sim)
    Area = get_area(cps_start .* sim.L)
    t0 = 0; a0 = 0; v0 = 0; p0 = 2*D

    @gif for tᵢ in range(t₀, t₀ + duration; step)
        t = sum(sim.flow.Δt[1:end-1])
        while t < tᵢ * sim.L / sim.U
            cps = cps_at_time(pathing, 105, t) .* D .+ SA{T}[2*D-0.1*t, 2.5D]
            sim.sim.body = ParametricBodies.update!(sim.sim.body, cps, sim.flow.Δt[end])
            sim_step!(sim, t / sim.L; remeasure = true)
            force = -WaterLily.pressure_force(sim) #/ (sim.L * 0.5)
            Δt = sim.flow.Δt[end]
            @show force, p0
            accel = (force[1] / Area)
            p0 += Δt * (v0 + Δt * accel / 2.)
            # @show force, p0
            push!(displacement, p0)
            v0 += Δt * accel
            push!(velocity, v0)
            a0 = accel
            push!(acceleration, a0)
            push!(timesteps, sim.flow.Δt[end])
            push!(forces, force[1])
            push!(time, t)
            push!(time_sim, sim_time(sim))
            for (i, p) in enumerate(cps[1, :])
                push!(cps_paths_x[i], p)
            end
            t0 = t; t += sim.flow.Δt[end]
        end
        force_out = -WaterLily.pressure_force(sim) / (sim.L * 0.5)
        push!(forces_out, force_out[1]) # plot
        R = inside(sim.flow.p)
        gen_p_plots(sim, t)
        gen_ω_gif(sim, t, R)
        println("tU/L=", round(tᵢ, digits = 4), ", Δt=", round(sim.flow.Δt[end], digits = 3))
    end

    diffs = [(cps_paths_x[25][i+1] - cps_paths_x[25][i]) / timesteps[i+1] for i in 1:length(cps_paths_x[25])-1]

    display(Plots.plot(diffs, xlabel = "numerical time", ylabel = "velocity", title = "Num. Velocity CP25", label = "cps_x 25"))
    display(Plots.plot(cps_paths_x[25]))
    display(Plots.plot(forces, xlabel="numerical time", ylabel="force", title="Pressure Force on Jellyfish"))
    display(Plots.plot(time))
    display(Plots.plot(time_sim))

    return forces, time, time_sim, timesteps, cps_paths_x, forces_out, displacement, velocity, acceleration
end

# forces, time, time_sim, timesteps, cps_paths_x, forces_out, displacement, velocity, acceleration = run_jelly_simulation(cps_start, D, Re, U, ϵ, thk, deg, pathing)
# # steps = 3 * duration / length(new_cps_list)
# # #duration= 15; t₀=round(sim_time(sim))
# # # time = range(t₀,t₀+duration; step=0.1)
# # # results = [get_forces!(sim, tᵢ, duration, new_cps_list; ThreeD=ThreeD) for tᵢ in time]
# plt = Plots.plot(new_cps_list[1][1,:] .* sim.L, new_cps_list[1][2,:] .* sim.L)
# for i in 2:10
#     Plots.plot!(new_cps_list[i][1,:] .* sim.L, new_cps_list[i][2,:] .* sim.L)
# end
# display(plt)

# time_var_forces = [-0.0, -0.0, -0.6248582294427081, -0.005061873864426886, -0.37442454755814936, -0.01605573341883826, -0.009236531964714079, -0.22161401202437503, 0.019368927307737316, 0.020002712081691243, 0.042705200323929604, 0.09307403688843863, 0.07772224653716239, 0.05685283314541367, 0.10501777004641227, 0.12250975391269825, 0.24952699974113385, 0.24139688881341925, 0.30425365131837623, 0.30295250124172757, 0.3331982204033288, 0.35788497607340286, 0.33265042931371563, 0.3664960755127822, 0.2903497526491336, 0.3350858612025692, 0.2654395552757536, 0.21092639946056585, 0.12118992919325056, 0.09072700576825277, -0.02251472057020272, -0.09051344900463443, -0.17420183853317128, -0.26655715183261375, -0.3361958247432747, -0.36944896246518855, -0.42248826069768697, -0.4453092361478359, -0.43906053457919114, -0.4495830269476784, -0.42923982163379826, -0.4089163663772837, -0.3280622522742078, -0.2709390846068003, -0.10290235562790784, -0.1314945005144797, 0.0458876246316795, -0.03525261407152236, 0.5066837316191197, -0.0429809828891905, 0.5199857063343001, 0.40702431580547227, 0.5233668702684042, 0.46247917703083097, 0.527653234209037, 0.5392361242249277, 0.5008517441500759, 0.4736033254759846, 0.5975013312479973, 0.2108425511582861, 0.15180435111923352, 0.15400901263966427, -0.060481720170621145, -0.21354211342224302, -0.24648592461466734, -0.31917506649849514, -0.2902177503539219, -0.5146960385497652, -0.35302197670359803, -0.5087807767034035, -0.34188222580030847, -0.47153882674218806, -0.3501925204898164, -0.2886822873949755, -0.24957927157219828, -0.20177407393993008, -0.12034150086756057, -0.07111462960627968, 0.025638115242644588, 0.06260159188066128, 0.18719980313207213, 0.2656391198127608, 0.27508398787629884, 0.34454713035084517, 0.4041958130321467, 0.40580039880487817, 0.9096987782875772, 0.5872171928056722, 0.33081978696414893, 0.25046728056586764, 0.22362933019177778]

# forces = getindex.(results, 1)
# a = getindex.(results, 2)
# v = getindex.(results, 3)
# s = getindex.(results, 4)
# # t = getindex.(results, 5)
# display(Plots.plot(time, forces, xlabel="Time", ylabel="Force", title="Force vs Time"))
# display(Plots.plot(time, a, xlabel="Time", ylabel="Acceleration", title="Acceleration vs Time"))
# display(Plots.plot(time, v, xlabel="Time", ylabel="Velocity", title="Velocity vs Time"))
# display(Plots.plot(time, s, xlabel="Time", ylabel="Position", title="Position vs Time"))
# t_scale = duration / length(new_cps_list)
# diff_xs = []
# for i in 1:length(path_x)-1
#     diff_x = (path_x[i+1] - path_x[i]) / t_scale
#     push!(diff_xs, diff_x)
# end
# display(Plots.plot(diff_xs, xlabel="frame number", ylabel="Velocity", title="Velocity CP25", label="cps_x 25"))

# WaterLily.logger("test_psolver")
# res = simulate_Jelly!(sim, cps_start; duration=9, period=period, step=0.1, remeasure=true, plotbody=false, ThreeD=ThreeD)
# plot_logger("test_psolver")
# savefig("psolver.png")


function visualize_sdf_3D(body; D=2^7, n=75, T=Float32, surface_only=true)
    xs = range(-D, D, length = n)
    ys = range(-D, D, length = n)
    zs = range(-D, D, length = n)

    φ = [sdf(body, SA[T(x), T(y), T(z)]) for x in xs, y in ys, z in zs]
    @show φ
    fig = Figure(; size = (900, 700))
    ax = Axis3(fig[1, 1], title = "Signed Distance Field")

    xside = xs[1] .. xs[end]           
    yside = ys[1] .. ys[end]
    zside = zs[1] .. zs[end]

    if surface_only
        GLMakie.contour!(ax, xside, yside, zside, φ;
            levels = [0.0],
            colormap = :plasma,
            transparency = true,
            alpha = 0.9
        )
    else
        GLMakie.volume!(ax, xside, yside, zside, φ;
            colormap = :algae,
            # colorrange = (-D/10, D/10),
            transparency = true,
            alpha = 0.75
        )
        GLMakie.contour!(ax, xside, yside, zside, φ;
            levels = [0.0],
            colormap = :plasma,
            transparency = true,
            alpha = 0.5
        )
    end
    GLMakie.xlims!(ax, xs[1], xs[end]); GLMakie.ylims!(ax, ys[1], ys[end]); GLMakie.zlims!(ax, zs[1], zs[end])
    fig
    # ADD save functionality 
end

revolve_map(x,t) = SA[x[1], hypot(x[2], x[3])]

# R = 1.0
# cps = SA_F32[
#     R   R   0  -R  -R  -R   0   R   R;
#     0   R   R   R   0  -R  -R  -R   0
# ] * D/2
# weights = SA_F32[1., √2/2, 1., √2/2, 1., √2/2, 1., √2/2, 1.]
# knots   = SA_F32[0,0,0, 1/4,1/4, 1/2,1/2, 3/4,3/4, 1,1,1]
# curve   = NurbsCurve(cps, knots, weights)
# sphere = ParametricBody(curve; map=revolve_map, ndims=3)

# cps_j = new_cps_list[1] * D/2
# degree = deg
# n_ctrl = size(cps_j, 2)
# weights_j = ones(T, n_ctrl)
# knots_j = T.(clamped_uniform_knots(degree, n_ctrl))
# curve_j = NurbsCurve(cps_j, knots_j, weights_j)
# body = ParametricBody(curve_j; map=revolve_map, ndims=3)

# visualize_sdf_3D(sim.body; D=D, n=50, surface_only=false)