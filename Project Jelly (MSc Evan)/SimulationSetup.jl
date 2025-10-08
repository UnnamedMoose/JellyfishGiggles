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
    contour!(sim.flow.σ[R]'|>Array;levels,lines, title=title)        # Plot signed distance function of body
    # plot!(sim.body.curve, shift=(0.5, 0.5), add_cp=true)
    # xs = range(0, 300, length=200)
    # ys = range(0, 300, length=200)
    # Z = [sdf(sim.body, SA[x, y]) for y in ys, x in xs]

    # heatmap(xs, ys, Z; color=:viridis, aspect_ratio=1, title="Signed Distance Field")
    # contour!(xs, ys, Z, levels=[0.0], linewidth=2, color=:green, title=title)  # Contour where sdf=0

    # heatmap(sim.flow.σ[R]', clim=(-0.1, 0.1), title=title)  # this shows small nonzero ghost blobs
end

@inline function dynamicSpline(::Type{T}=Float32; 
    new_cps_list, D=2^7, Re=302, U=1, ϵ=0.5, thk=2ϵ+√3, deg, 
    mem=Array, use_biotsavart=false, U_func=nothing) where {T<:AbstractFloat}

    cps = new_cps_list[1] .* 2 .* D .+ SA{T}[3D, 3D]
    degree = deg
    n_ctrl = size(cps, 2)
    weights = ones(T, n_ctrl)
    knots = T.(clamped_uniform_knots(degree, n_ctrl))
    curve = NurbsCurve(cps, knots, weights)

    body = DynamicNurbsBody(curve; thk=thk, boundary=true)

    ν = U * D / Re

    # Wrap U_func into uBC(i, x, t) for WaterLily if provided
    # uBC = if U_func === nothing
    #     (U, 0)
    # else
    #     (i, x, t) -> U_func(x, t, D)[i]
    # end
    uBC = (0,0)
    return use_biotsavart ?
        BiotSimulation((8D, 6D), uBC, D; U, ν, body, T, mem, ϵ) :
        Simulation((8D, 6D), uBC, D; U, ν, body, T, mem, ϵ)
end

function sim_gif_forces!(sim, new_cps_list;
                         duration=1, period=3, step=0.1, verbose=true,
                         R=inside(sim.flow.p), remeasure=false, plotbody=false, kv...)

    Tp = eltype(sim.flow.p)
    t₀ = round(sim_time(sim))
    t = sum(sim.flow.Δt[1:end-1])  # current sim time

    v = Float32(0); s = Float32(0)
    in_period = true
    periodic_force = zero(Tp)
    t_start = 0

    # --- storage for force history ---
    f_hist = Tp[0]      # store Fx or full force vector if needed
    period = period * sim.L / sim.U

    anim = @animate for tᵢ in range(t₀, t₀+duration; step)
        while t < tᵢ * sim.L / sim.U
            sim.flow.Δt[end] = WaterLily.CFL(sim.flow; Δt_max=Tp(0.1))
            # Δt = sim.flow.Δt[end]
            cps_interp, v, s = interpolate_cps_hermite_new(new_cps_list, t, period, v, s, sim.flow.Δt[end], f_hist[end])
            @show v, s, f_hist[end]

            body_interpolation = cps_interp .* 2 .* sim.L .+ (Tp(3sim.L), Tp(3sim.L))

            sim.sim.body = ParametricBodies.update!(sim.sim.body, body_interpolation, sim.flow.Δt[end])

            sim_step!(sim, tᵢ; remeasure)

            raw    = WaterLily.total_force(sim)
            scaled = raw ./ (0.5 * sim.L * sim.U^2)
            if in_period
                periodic_force += scaled[1]
                if t - t_start >= period
                    in_period = false
                end
            end

            push!(f_hist, scaled[1])  # store x-force (or push!(f_hist, scaled) to keep both)
            t += sim.flow.Δt[end]'
        end
        # --- visualization ---
        @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L/sim.U
        @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I]) < 0.001, 0.0, sim.flow.σ[I])
        # @inside sim.flow.σ[I] = WaterLily.div(I,sim.flow.u)
        # @show maximum(abs,sim.flow.σ[R]|>Array)
        # push!(div, maximum(abs,sim.flow.σ[R]|>Array))
        flood(sim.flow.σ[R] |> Array; clims=(-5,5), kv...)
        # contour(sim.flow.p')
        plotbody && body_plot!(sim; title="$tᵢ")

        verbose && println("t=", round(t, digits=4),
                           ", Δt=", round(sim.flow.Δt[end], digits=3))
    end 

    gif(anim,"Swimming_Jelly.gif")

    return (forces=f_hist)
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


