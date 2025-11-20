### Current Functions
function clamped_uniform_knots(p::Int, Ncp::Int)
    @assert Ncp > p "Need at least p+1 control points"
    Ninterior = Ncp - p - 1
    # start: p+1 zeros
    head = zeros(Float64, p + 1)
    # interior: strictly between (0,1), uniform
    interior = Ninterior > 0 ? collect(range(0.0, 1.0, length = Ninterior + 2))[2:end-1] : Float64[]
    # end: p+1 ones
    tail = ones(Float64, p + 1)
    return vcat(head, interior, tail)
end

function get_body!(bod,sim,t=WaterLily.time(sim))
    @inside sim.flow.σ[I] = WaterLily.sdf(sim.body,SVector(Tuple(I).-0.5f0),t)
    copyto!(bod,sim.flow.σ[inside(sim.flow.σ)])
end

addbody(x,y;c=:black) = Plots.plot!(Shape(x,y), c=c, legend=false)
function body_plot!(sim;levels=[0],lines=:black,R=inside(sim.flow.p),title)
    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
    Plots.contour!(sim.flow.σ[R]'|>Array;levels,lines, title=title)        # Plot signed distance function of body
    # plot!(sim.body.curve, shift=(0.5, 0.5), add_cp=true)
    # xs = range(0, 300, length=200)
    # ys = range(0, 300, length=200)
    # Z = [sdf(sim.body, SA[x, y]) for y in ys, x in xs]

    # heatmap(xs, ys, Z; color=:viridis, aspect_ratio=1, title="Signed Distance Field")
    # contour!(xs, ys, Z, levels=[0.0], linewidth=2, color=:green, title=title)  # Contour where sdf=0

    # heatmap(sim.flow.σ[R]', clim=(-0.1, 0.1), title=title)  # this shows small nonzero ghost blobs
end

function smoothstep(x, edge0, edge1)
    t = clamp((x - edge0) / (edge1 - edge0), 0.0f0, 1.0f0)
    return t * t * (3f0 - 2f0 * t)
end

function U_func(x, t, L)
    const_v = 0.2f0
    α = smoothstep(t, 47.0f0, 49.0f0)  # Blend window around x = 48
    vx = (1f0 - α)*v + (α* const_v)
    return (v, 0.0f0)
end

@inline function TwoDimJellyfish(::Type{T}=Float32; 
    new_cps_list, D=2^7, Re=302, U=1, ϵ=0.5, thk=2ϵ+√3, deg, 
    mem=Array, use_biotsavart=false, U_func=nothing) where {T<:AbstractFloat}

    cps = new_cps_list[1] .* 1 .* D .+ SA{T}[D, 2D]
    degree = deg
    n_ctrl = size(cps, 2)
    weights = ones(T, n_ctrl)
    knots = T.(clamped_uniform_knots(degree, n_ctrl))
    curve = NurbsCurve(cps, knots, weights)
    
    # function Mapping(x,t)
    #     x - SA[x₀ + (t - t₀) * vx₀, 0]
    # end

    body = DynamicNurbsBody(curve; thk=thk, boundary=true)
    # body = AutoBody((x,t)->curve(s), Mapping(x,t))

    ν = U * D / Re

    # Wrap U_func into uBC(i, x, t) for WaterLily if provided
    uBC = if U_func === nothing
        (0, 0)
    else
        (i, x, t) -> U_func(x, t, D)[i]
    end
    # uBC = (0,0)
    return use_biotsavart ?
        BiotSimulation((4D, 4D), (0,0), D; U, ν, body, T, mem, ϵ) :
        Simulation((4D, 4D), (0,0), D; U, ν, body, T, mem, ϵ)
end

function simulate_Jelly!(sim, new_cps_list;
                         duration=1, period=3, step=0.1, verbose=true,
                         R=inside(sim.flow.p), remeasure=false, plotbody=false, kv...)

    Tp = eltype(sim.flow.p)
    t₀ = round(sim_time(sim))
    t = sum(sim.flow.Δt[1:end-1])

    v = Float32(0)
    s = Float32(0)
    Area = get_area(new_cps_list[1]) * sim.L^2

    forces = Tp[0]   
    vel = Tp[0]      
    crit_sets = []  
    period = period * sim.L / sim.U
    time = [0.0, period/10, 2period/10, 3period/10, 4period/10, 5period/10, 6period/10, 7period/10, 8period/10, 9period/10]
    Δt_max = Tp(0.5)
    grow_cap = Tp(1.1)

    anim = @animate for tᵢ in range(t₀, t₀+duration; step)
        while t < tᵢ * sim.L / sim.U
            # sim.flow.Δt[end] = WaterLily.CFL(sim.flow; Δt_max=Tp(0.1))
            # cps_interp= interpolate_cps_hermite_new(new_cps_list, t, period)
            cps_interp = cps_fourier_interpolator(new_cps_list, time, t, period, Int(5))
            body_interpolation = cps_interp .* 1 .* sim.L .+ (Tp(sim.L), Tp(2sim.L))

            sim.sim.body = ParametricBodies.update!(sim.sim.body, body_interpolation, sim.flow.Δt[end])

            sim_step!(sim, tᵢ; remeasure)

            # push!(vel, v)

            t₀ = t; t += sim.flow.Δt[end]
        end
        raw    = -WaterLily.total_force(sim)
        scaled = raw ./ (0.5 * sim.L * sim.U^2) 
        # if abs(scaled[1]) > 5
        #         push!(crit_sets, cps_interp)
        # end

        # a = (scaled[1])/(Area)
        # v += sim.flow.Δt[end]*a
        # s += sim.flow.Δt[end]*(v+sim.flow.Δt[end]*a/2.)

        push!(forces, scaled[1]) 

        @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L/sim.U
        @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I]) < 0.001, 0.0, sim.flow.σ[I])
        flood(sim.flow.σ[R] |> Array; clims=(-5,5), kv...)
        # speed = @. sqrt(sim.flow.u[:,:,1]^2 + sim.flow.u[:,:,2]^2)
        # flood(speed |> Array; camp=:algae, clims=(0, sim.U), kv...)
        plotbody && body_plot!(sim; title="$tᵢ")

        verbose && println("t=", round(t, digits=4),
                           ", Δt=", round(sim.flow.Δt[end], digits=3))
    end 

    gif(anim,"Swimming_Jelly.gif")

    return (forces=forces, vel=vel)
end

function simulate_Jelly_Fourier!(sim, new_cps_list;
    duration=1, period=3, step=0.1, verbose=true,
    R=inside(sim.flow.p), remeasure=false, plotbody=false, kv...)

    Tp = eltype(sim.flow.p)
    t₀ = round(sim_time(sim))
    t  = sum(sim.flow.Δt[1:end-1])
    L, U = sim.L, sim.U

    forces = Tp[0]
    p_locs = []
    pressures = []
    raw = Tp(0)
    crit_sets = []
    period = period * L / U
    time = [i*period/10 for i in 0:9]

    # Fourier model (built once)
    model = build_cps_fourier(new_cps_list, time, period, K=5)
    @show model

    Δx = 1
    # ε  = 2.0 * Δx              # keep ε/Δx constant
    GEOM_TOL = 0.01 * Δx       # motion threshold
    SUB_STEPS = 2              # geometry sub-steps

    cps_prev, _, _ = eval_cps(model, t)

    anim = @animate for tᵢ in range(t₀, t₀ + duration; step)
        while t < tᵢ * L / U
            sim.flow.Δt[end] = WaterLily.CFL(sim.flow; Δt_max=Tp(0.01))

            # --- mid-point geometry update
            t_mid = t + 0.5 * sim.flow.Δt[end]
            cps_mid, dcps_mid, _ = eval_cps(model, t_mid)
            # @show cps_mid

            # scale + shift
            body_interpolation = cps_mid .* 1 .* L .+ (Tp(sim.L), Tp(2sim.L))

            # check if body moved significantly
            Δ = maximum(norm.(eachcol(cps_mid .- cps_prev)))
            # @show Δ
            if Δ > GEOM_TOL
                # sub-step geometry update
                for m in 1:SUB_STEPS
                    t_sub = t + (m - 0.5) * sim.flow.Δt[end] / SUB_STEPS
                    cps_sub, _, _ = eval_cps(model, t_sub)
                    body_sub = cps_sub .* 1 .* L .+ (Tp(sim.L), Tp(2sim.L))
                    sim.sim.body = ParametricBodies.update!(sim.sim.body, body_sub, sim.flow.Δt[end] / SUB_STEPS)
                end
                cps_prev = cps_mid
            end

            # --- main CFD step
            sim_step!(sim, tᵢ; remeasure)

            # # --- forces
            # raw = -WaterLily.total_force(sim)
            # scaled = raw ./ (0.5 * L * U^2)
            # push!(forces, scaled[1])

            # if abs(scaled[1]) > 5
            #     push!(crit_sets, cps_mid)
            # end

            t₀ = t
            t += sim.flow.Δt[end]
        end
        raw    = -WaterLily.total_force(sim)[1] #* (step * sim.L / sim.U)
        scaled = raw ./ (0.5 * sim.L * sim.U^2) 
        # if abs(scaled[1]) > 5
        #         push!(crit_sets, cps_interp)
        # end

        # a = (scaled[1])/(Area)
        # v += sim.flow.Δt[end]*a
        # s += sim.flow.Δt[end]*(v+sim.flow.Δt[end]*a/2.)

        save_dir = joinpath(pwd(), "Normals_check")
        isdir(save_dir) || mkpath(save_dir)

        x = range(0, 130; length=130)
        y = range(0, 130; length=130)
        nx, ny = length(x), length(y)

        # Arrays for plotting (only nonzero vectors)
        xs, ys, nxs, nys = Float64[], Float64[], Float64[], Float64[]
        p_masked = fill(NaN, nx, ny)
        for j in 1:ny, i in 1:nx
            nvec = WaterLily.nds(sim.body, SVector(x[i], y[j]), 0.0)
            if norm(nvec) > 1e-6       # skip zero (or near-zero) vectors
                push!(xs, x[i])
                push!(ys, y[j])
                push!(nxs, nvec[1])
                push!(nys, nvec[2])
                p_masked[i,j] = sim.flow.p[i,j]
            end
        end

        fig = Figure(resolution=(700,700))
        ax = Axis(fig[1,1], title="Surface Normals (nds)", aspect=DataAspect())

        arrows!(ax, xs, ys, nxs, nys, arrowsize=10, lengthscale=3, color=:blue)
        save(joinpath(save_dir, "nds_frame_$(tᵢ).png"))
        # save("nds_frame_$(tᵢ).png", fig)

        save_dir_p = joinpath(pwd(), "Pressure_check")
        isdir(save_dir_p) || mkpath(save_dir_p)
        # p = sim.flow.p
        # push!(p_locs, argmax(p_masked))
        p_sum = sum(sim.flow.p)
        push!(pressures, p_sum)
        # Get indices of non-NaN cells
        valid_inds = findall(!isnan, p_masked)

        # Extract valid values
        valid_vals = p_masked[valid_inds]

        # Find local extrema ignoring NaNs
        imax = argmax(valid_vals)
        imin = argmin(valid_vals)

        # Map back to full-array indices
        idx_max = valid_inds[imax]
        idx_min = valid_inds[imin]

        @show idx_max, p_masked[idx_max]
        @show idx_min, p_masked[idx_min]
        @show mean(p_masked[valid_inds])
        pressure_plot = Plots.heatmap(p_masked', aspect_ratio=1,
        c=:balance,          # diverging colormap (blue–white–red)
        clims=(-100, 100),  # symmetric limits
        title="Pressure Field (white = 0)")
        WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
        Plots.contour!(sim.flow.σ[inside(sim.flow.p)]'|>Array;levels=[0],lines=:black)
        savefig(pressure_plot, joinpath(save_dir_p, "pressure_$(tᵢ).png"))

        push!(forces, scaled) 
        # --- plotting
        @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * L/U
        @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I]) < 0.001, 0.0, sim.flow.σ[I])
        # @inside sim.flow.σ[I] = WaterLily.div(I,sim.flow.u)
        # @show maximum(abs,sim.flow.σ[R]|>Array)
        # push!(div, maximum(abs,sim.flow.σ[R]|>Array))
        flood(sim.flow.σ[R] |> Array; clims=(-0.5,0.5), kv...)

        plotbody && body_plot!(sim; title="$(round(t, digits=4))")

        verbose && println("t=", round(t, digits=4), ", Δt=", round(sim.flow.Δt[end], digits=3))

    end

    gif(anim, "Swimming_Jelly.gif")
    return (forces=forces, pressures = pressures)
end




@inline function ThreeDimJellyfish(::Type{T}=Float32; 
    new_cps_list, D=2^7, Re=302, U=1, ϵ=0.5, thk=2ϵ+√3, deg, 
    mem=Array, use_biotsavart=false, U_func=nothing) where {T<:AbstractFloat}
    @show typeof(new_cps_list[1])
    cps = new_cps_list[1] .* 1 .* D .+ SA{T}[D, 2D, D/2]
    degree = deg
    n_ctrl = size(cps, 2)
    weights = ones(T, n_ctrl)
    knots = T.(clamped_uniform_knots(degree, n_ctrl))
    curve = NurbsCurve(cps, knots, weights)
    
    map(p, θ) = SVector(p[1], p[2]*cos(θ), p[2]*sin(θ))

    body = DynamicNurbsBody(curve; thk=thk, boundary=false)

    ν = U * D / Re

    return use_biotsavart ?
        BiotSimulation((4D, 4D, D), (0,0,0), D; U, ν, body, T, mem, ϵ) :
        Simulation((4D, 4D, D), (0,0,0), D; U, ν, body, T, mem, ϵ)
end














### Older functions

function sim_gif!(sim;duration=1,step=0.1,verbose=true,R=inside(sim.flow.p),
                  remeasure=false,plotbody=false,kv...)
    t₀ = round(sim_time(sim))
    # @show t₀
    # t₀ = 0
    t = sum(sim.flow.Δt[1:end-1])
    v = Float32(0); s = Float32(0)
    period = Tp(3) * sim.L / sim.U
    in_period = true
    periodic_force = zero(Tp)
    t_start = 0
    
    @time @gif for tᵢ in range(t₀,t₀+duration;step)
        while t < tᵢ * sim.L / sim.U
            sim.flow.Δt[end] = WaterLily.CFL(sim.flow; Δt_max=Tp(0.1))
            Δt = sim.flow.Δt[end]

            # push!(sim.flow.Δt, Δt)
            # Δt = sim.flow.Δt[end]
            interpolated, v, s = interpolate_cps_hermite(new_cps_list, t, Δt, sim, v, s, periodic_force)
            interpolated = SMatrix{2,41,Float32,82}(interpolated)

            sim.sim.body = ParametricBodies.update!(sim.sim.body, interpolated, Δt)

            
            # @show sim.sim.body
            sim_step!(sim,tᵢ;remeasure)

            raw    = WaterLily.total_force(sim)
            scaled = raw ./ (0.5 * sim.L * sim.U^2)
            @show scaled

            if in_period
                periodic_force += scaled[1]
                if t - t_start >= period
                    in_period = false
                end
                # return periodic_force
            end
            t += Δt
        end
        @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I])<0.001,0.0,sim.flow.σ[I])
        flood(sim.flow.σ[R]|>Array; clims=(-5,5), kv...)
        plotbody && body_plot!(sim)
        verbose && println("t=",round(t,digits=4),
                           ", Δt=",round(sim.flow.Δt[end],digits=3))
    end
end

function sim_frames!(sim;
    duration=15, step=0.1, verbose=true,
    R = inside(sim.flow.p),
    remeasure=false, plotbody=false,
    outdir = ".", prefix = "frame", ext = ".png",
    clims=(-5,5), kv...
)
    t₀ = round(sim_time(sim))               # same as your GIF
    t  = sum(sim.flow.Δt[1:end-1])          # cumulative physical time used in while
    v = Float32(0); s = Float32(0)
    times = [Tp(3), Tp(3+3/5), Tp(3+6/5), Tp(3+9/5), Tp(3+12/5), Tp(3+15/5)]

    period = Tp(3) * sim.L / sim.U
    in_period = true
    periodic_force = zero(Tp)
    t_start = 0

    scale = sim.L / sim.U                   # nondim → physical time

    n = 0
    for tᵢ in range(t₀, t₀ + duration; step)
        tᵢ_phys = tᵢ * scale

        # advance solver until we reach this frame’s target time
        while t < tᵢ_phys
            Δt = sim.flow.Δt[end]

            interpolated, v, s = interpolate_cps_hermite(new_cps_list, t, Δt, sim, v, s, periodic_force)
            interpolated = SMatrix{2,41,Float32,82}(interpolated)
            sim.sim.body = ParametricBodies.update!(sim.sim.body, interpolated, Δt)

            # IMPORTANT: same as your GIF — pass the frame time, not the integrator time
            sim_step!(sim, tᵢ; remeasure)

            raw    = WaterLily.total_force(sim)
            scaled = raw ./ (0.5 * sim.L * sim.U^2)

            if in_period
                periodic_force += scaled[1]
                if t - t_start >= period
                    in_period = false
                end
            end

            t += Δt
        end
        @show tᵢ
        if Float32(tᵢ) in times
            # render this moment
            @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L / sim.U
            @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I]) < 0.001, 0.0, sim.flow.σ[I])

            flood(sim.flow.σ[R] |> Array; clims=clims, kv...)
            # plotbody && body_plot!(sim)

            WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
            # contour!(sim.flow.σ[R]'|>Array;levels,lines)
            plot!(sim.body.curve, title = "Timestep = $(tᵢ)", xlabel = "Lx" , ylabel = "Ly", shift=(0.5, 0.5), add_cp=true)

            fn = joinpath(outdir, "$(prefix)_$(lpad(n,4,'0'))_t$(round(tᵢ; digits=4))$(ext)")
            savefig(fn)
        end

        verbose && println("t=", round(t, digits=4),
                           ", Δt=", round(sim.flow.Δt[end], digits=3))
    end

    return nothing
end

"""
    make_circle_cps(T::Type, npoints::Int, nsteps::Int; radius=1.0, shift_per_step=(0.05,0.02))

Create a list of control point sets for a slowly moving circle.
Each element is an `SMatrix{2,npoints,T}` containing [x; y] control points.
"""

function make_circle_cps(T::Type, npoints::Int, nsteps::Int; radius=1.0, shift_per_step=(0.05,0.02))
    cps_list = Vector{SMatrix{2,npoints,T}}(undef, nsteps)

    θ = range(0, 2π, length=npoints)  # circle parameterization

    for k in 0:nsteps-1
        shift = (k * shift_per_step[1], k * shift_per_step[2])  # slow movement
        x = radius .* cos.(θ) .+ shift[1]
        y = radius .* sin.(θ) .+ shift[2]
        cps_list[k+1] = SMatrix{2,npoints,T}(vcat(x', y')...)  # row-major constructor
    end

    return cps_list
end

function make_circle_cps_sin_motion(T::Type, npoints::Int, nsteps::Int; radius=1.0, freq=0.1, amplitude=(1.0, 1.0))
    cps_list = Vector{SMatrix{2, npoints, T}}(undef, nsteps)

    θ = range(0, 2π, length=npoints)

    for k in 0:nsteps-1
        shift = (amplitude[1] * sin(2π * freq * k), amplitude[2] * cos(2π * freq * k))
        x = radius .* cos.(θ) .+ shift[1]
        y = radius .* sin.(θ) .+ shift[2]
        cps_list[k+1] = SMatrix{2, npoints, T}(vcat(x', y')...)
    end

    return cps_list
end

function sim_gif_2!(sim, new_cps_list;duration=1,step=0.1,verbose=true,R=inside(sim.flow.p),
                  remeasure=false,plotbody=false,kv...)
    t₀ = round(sim_time(sim))
    # @show t₀
    # t₀ = 0
    t = sum(sim.flow.Δt[1:end-1])
    v = Float32(0); s = Float32(0)
    period = Tp(3) * sim.L / sim.U
    in_period = true
    periodic_force = zero(Tp)
    t_start = 0

    Δt_cps = 0.05   # spacing of your cps list (seconds)
    nsteps  = length(new_cps_list)
    
    @time @gif for tᵢ in range(t₀,t₀+duration;step)
        while t < tᵢ * sim.L / sim.U
            sim.flow.Δt[end] = WaterLily.CFL(sim.flow; Δt_max=Tp(0.1))
            Δt = sim.flow.Δt[end]
            # idx = clamp(round(Int, (t/Δt_cps) + 1), 1, nsteps)
            t_phys = t * sim.U / sim.L
            k = floor(Int, t_phys/Δt_cps) + 1
            # @show k
            α = (t_phys % Δt_cps) / Δt_cps
            cps_interp = (1-α) .* new_cps_list[k] .+ α .* new_cps_list[min(k+1, nsteps)]
            body_interpolation = cps_interp .* sim.L .+ (Tp(sim.L), Tp(2sim.L))

            body_interpolation = SMatrix{2,41,Float32,82}(body_interpolation)
            # body_interpolation = new_cps_list[idx] .* sim.L .+ (Tp(4sim.L),Tp(3sim.L))

            sim.sim.body = ParametricBodies.update!(sim.sim.body, body_interpolation, Δt)

            sim_step!(sim,tᵢ;remeasure)

            # raw    = WaterLily.total_force(sim)
            # scaled = raw ./ (0.5 * sim.L * sim.U^2)
            # @show scaled

            # if in_period
            #     periodic_force += scaled[1]
            #     if t - t_start >= period
            #         in_period = false
            #     end
            #     # return periodic_force
            # end
            t += Δt
        end

        raw = WaterLily.total_force(sim)
        scaled = raw ./ (0.5 * sim.L * sim.U^2)

        @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I])<0.001,0.0,sim.flow.σ[I])
        flood(sim.flow.σ[R]|>Array; clims=(-5,5), kv...)
        plotbody && body_plot!(sim)
        verbose && println("t=",round(t,digits=4),
                           ", Δt=",round(sim.flow.Δt[end],digits=3))
    end
end


