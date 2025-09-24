function clamped_uniform_knots(p::Int, n::Int)
    k = zeros(Float64, n + p + 1)
    k[p+1:end-p] .= range(0, 1, length=n - p + 1)
    k[end-p+1:end] .= 1.0
    return k
end

# grid_size = 48, meaning that I want the jellyfish diameter to span 48 grid cells.
# Should I then scale all control points with the actual D_act = 1.25 meter before inputting them to the simulation?
# D_act / grid_size = 1.25 / 48 = 0.02604166667 cm per grid cell.
# cps * grid_size means that the jellyfish will cover only 1 grid cell ( I think??)
# 

@inline function dynamicSpline(::Type{T}=Float32; new_cps_list,D=2^7,Re=302,U=1,ϵ=0.5,thk=2ϵ+√3,mem=Array, use_biotsavart=false) where {T<:AbstractFloat}
    cps = new_cps_list[1] .* D .+ SA{T}[4D,3D]
    degree = 2
    n_ctrl = size(cps, 2)
    weights = ones(T, n_ctrl)
    knots = T.(clamped_uniform_knots(degree, n_ctrl))

    curve = NurbsCurve(cps, knots, weights)         

    body = DynamicNurbsBody(curve; thk=thk, boundary=true)
    ν = U*D/Re
    return use_biotsavart ?
    BiotSimulation((8D,6D),(0,0),D; U, ν, body, T, mem, ϵ) :
    Simulation((8D,6D),(0,0),D; U, ν, body, T, mem,ϵ, 
    # exitBC=true   
    )
end

# @inline function dynamicSpline(::Type{T}=Float32; new_cps_list,L=2^5,Re=302,U=1,ϵ=0.5,thk=2ϵ+√3,mem=Array, use_biotsavart=false) where {T<:AbstractFloat} 
#     cps = new_cps_list[1] .* L .+ SA{T}[4L,3L] 
#     degree = 2 
#     n_ctrl = size(cps, 2) 
#     weights = ones(T, n_ctrl) 
#     knots = T.(clamped_uniform_knots(degree, n_ctrl)) 
#     curve = NurbsCurve(cps, knots, weights) 
#     body = DynamicNurbsBody(curve; thk=thk, boundary=true) 
#     ν = U*L/Re 
#     return use_biotsavart ? 
#     BiotSimulation((8L,6L),(0,0),L; U, ν, body, T, mem, ϵ) : 
#     Simulation((8L,6L),(0,0),L; U, ν, body, T, mem,ϵ, # exitBC=true 
#     ) 
# end

function get_body!(bod,sim,t=WaterLily.time(sim))
    @inside sim.flow.σ[I] = WaterLily.sdf(sim.body,SVector(Tuple(I).-0.5f0),t)
    copyto!(bod,sim.flow.σ[inside(sim.flow.σ)])
end

addbody(x,y;c=:black) = Plots.plot!(Shape(x,y), c=c, legend=false)
function body_plot!(sim;levels=[0],lines=:black,R=inside(sim.flow.p))
    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
    # contour!(sim.flow.σ[R]'|>Array;levels,lines)
    plot!(sim.body.curve, shift=(0.5, 0.5), add_cp=true)
end

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
            body_interpolation = cps_interp .* sim.L .+ (Tp(4sim.L), Tp(3sim.L))

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

function sim_gif_forces!(sim, new_cps_list;
                         duration=1, period=3, cycles=1, step=0.1, verbose=true,
                         R=inside(sim.flow.p), remeasure=false, plotbody=false, kv...)

    Tp = eltype(sim.flow.p)
    t₀ = round(sim_time(sim))
    t = sum(sim.flow.Δt[1:end-1])  # current sim time

    v = Float32(0); s = Float32(0)
    in_period = true
    periodic_force = zero(Tp)
    t_start = 0

    # --- storage for force history ---
    ts   = Tp[]
    dts  = Tp[]
    f_hist = Tp[]      # store Fx or full force vector if needed
    interp_state = Tp[]
    div = Tp[]

    T      = period * sim.L / sim.U  # total period length
    nsteps = Int(length(new_cps_list) / cycles)
    @show typeof(nsteps)
    Δt_cps = period / (nsteps-1) * sim.L / sim.U  # spacing of cps list (seconds)

    anim = @animate for tᵢ in range(t₀, t₀+duration; step)
        while t < tᵢ * sim.L / sim.U
            # adaptive timestep
            sim.flow.Δt[end] = WaterLily.CFL(sim.flow; Δt_max=Tp(0.1))
            Δt = sim.flow.Δt[end]

            # --- geometry interpolation ---
            τ = mod(t, T)
            k = floor(Int, t / Δt_cps)+1

            α = (τ % Δt_cps) / Δt_cps

            cps_k  = new_cps_list[k]
            cps_k1 = new_cps_list[mod1(k+1, nsteps)]
            cps_interp = (1-α) .* cps_k .+ α .* cps_k1
            
            # @show new_cps_list[k]
            body_interpolation = cps_interp .* sim.L .+ (Tp(4sim.L), Tp(3sim.L))
            body_interpolation = SMatrix{2, size(cps_k,2), Float32}(body_interpolation)

            sim.sim.body = ParametricBodies.update!(sim.sim.body, body_interpolation, Δt)

            # --- advance one step ---
            sim_step!(sim, tᵢ; remeasure)

            # verbose && @show scaled
            # push!(interp_state, t)

            # if in_period
            #     periodic_force += scaled[1]
            #     if t - t_start >= period
            #         in_period = false
            #     end
            # end
            # raw    = WaterLily.total_force(sim)
            # scaled = raw ./ (0.5 * sim.L * sim.U^2)
            # push!(f_hist, scaled[1])  # store x-force (or push!(f_hist, scaled) to keep both)
            # push!(ts, t)
            t += Δt
        end
        # --- forces ---

        # push!(dts, sim.flow.Δt[end])

        # --- visualization ---
        @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L/sim.U
        @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I]) < 0.001, 0.0, sim.flow.σ[I])
        # @inside sim.flow.σ[I] = WaterLily.div(I,sim.flow.u)
        # @show maximum(abs,sim.flow.σ[R]|>Array)
        # push!(div, maximum(abs,sim.flow.σ[R]|>Array))
        flood(sim.flow.σ[R] |> Array; clims=(-0.05,0.05), kv...)
        plotbody && body_plot!(sim)

        verbose && println("t=", round(t, digits=4),
                           ", Δt=", round(sim.flow.Δt[end], digits=3))
    end 

    gif(anim,"Swimming_Jelly.gif")

    return (ts=ts, dts=dts, forces=f_hist,
            periodic_force=periodic_force,
            interp_state=interp_state, div=div)
end
