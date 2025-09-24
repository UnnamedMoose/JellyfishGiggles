function get_cps_scatter(new_cps_list)
    n = length(new_cps_list)
    plot_layout = @layout [grid(ceil(Int,n/2),2)]
    plt = plot(layout = plot_layout, size = (600, 200*n))

    for (i, cps) in enumerate(new_cps_list)
        plot!(plt, cps[1, :], cps[2, :],
            seriestype = :scatter,
            aspect_ratio = 1,
            xlims = (-0.25, 2.5),
            ylims = (-1.5, 1.5),
            title = "Control Points $i",
            xlabel = "x",
            ylabel = "y",
            legend = false,
            subplot = i
        )
    end

    display(plt)
end

function get_curves(crvs)
    c = length(crvs)
    plot_layout = @layout [grid(ceil(Int,c/2),2)]
    plt = plot(layout = plot_layout, size = (600, 200*n))

    for (i, crv) in enumerate(crvs)
        plot!(plt, crv,
            seriestype = :scatter,
            aspect_ratio = 1,
            xlims = (-0.25, 2.5),
            ylims = (-1.5, 1.5),
            title = "Control Points $i",
            xlabel = "x",
            ylabel = "y",
            legend = false,
            subplot = i
        )
    end
    savefig(plt, "interp_steps.png")
    display(plt)
end

function get_shape_error(areas, rel_errors, rel_area)
    println("Shape Area Summary:")
    println("────────────────────────────────────────────")
    @printf("%-10s %-15s %-15s\n", "Curve t/T", "Area", "Δ% from t = 0")
    println("────────────────────────────────────────────")
    n = length(areas)
    for i in 1:n
        @printf("Curve %-4d  %-15.6f %-15.2f\n", i, areas[i], rel_errors[i])
    end

    # plot(1:length(areas), areas; label="Area", xlabel="t/T", ylabel="Area", linewidth=2, legend=:topleft, title="Area per time step")
    plt = plot(1:length(rel_area), rel_area; label="Relative Area [-]", xlabel="t/T [-]", ylabel="Relative Area [-]",  linewidth=2, legend=:false, linestyle=:dash, title="Relative Area per time step")
    display(plt)
end

function poly_area(points::Vector{SVector{2,T}}) where T # Polygon area using the shoelace formula
    n = length(points)
    sum = zero(T)
    for i in 1:n
        x1, y1 = points[i]
        x2, y2 = points[mod1(i+1, n)]
        sum += x1 * y2 - x2 * y1
    end
    return abs(sum) / 2
end

function get_centroid(cps::SMatrix{2,41,T}) where {T}
    idx = SMatrix{2,18,T}(hcat(cps[:,13:29],cps[:,13]))
    idx2 = cps[2,13:21]
    centroid = Float32(abs(sum(idx2)) / length(idx2))  
    return idx, centroid
end

function get_area(cps)                          
    curve = BSplineCurve(cps; degree=2)
    points = [curve(s) for s in s_vals]
    area = poly_area(points)
    return area
end

function compute_diffs(new_cps_list)
    n = length(new_cps_list)
    diff = Vector{Float32}(undef, n)
    for i in 1:n
        if i == 1
            diff[i] = mean(new_cps_list[1] .- new_cps_list[n])
        else
            diff[i] = mean(new_cps_list[i] .- new_cps_list[i-1])
        end
    end
    return diff
end

function get_forces!(sim; duration=1,step=0.1,verbose=true)
    Tp = eltype(sim.flow.p)
    # current solver time (use whatever you consider the ground truth; or track it yourself)
    t₀ = round(sim_time(sim))
    t = sum(sim.flow.Δt[1:end-1])  # note: includes last step; adjust if you meant [1:end-1]
    v = Float32(0); s = Float32(0)
    period = Tp(3) * sim.L / sim.U

    ts  = Tp[]           # time history
    dts = Tp[]           # Δt history
    f_hist = Tp[]  # (Fx,Fy) per step, optional
    interp_state = Tp[]

    in_period = true
    periodic_force = zero(Tp)
    t_start = 0

    @time for tᵢ in range(t₀, t₀ + duration; step)
        while t < tᵢ * sim.L / sim.U
            # Δt = sim.flow.Δt[end]
            sim.flow.Δt[end] = WaterLily.CFL(sim.flow; Δt_max=Tp(0.1))
            Δt = sim.flow.Δt[end]
            push!(ts,  t)
            push!(dts, Δt)

            # update geometry
            interpolated, v, s, interp = interpolate_cps_hermite(new_cps_list, t, Δt, sim, v, s, periodic_force)
            push!(interp_state, interp)
            # interpolated = SMatrix{2,41,Float32,82}(interpolated)
            sim.sim.body = ParametricBodies.update!(sim.sim.body, interpolated, Δt)
            
            # advance one step
            sim_step!(sim, tᵢ; remeasure=true)
            raw    = WaterLily.total_force(sim)
            scaled = raw ./ (0.5 * sim.L * sim.U^2)
            @show scaled
            push!(f_hist, scaled[1])
            # if in_period
            #     periodic_force += scaled[1]
            #     if t - t_start >= period
            #         in_period = false
            #     end
            # end

            t += Δt
        end

        verbose && println("t=",round(t,digits=4),
                           ", Δt=",round(sim.flow.Δt[end],digits=3))

    end
    return ( ts = ts, dts = dts, forces = f_hist, periodic_force = periodic_force, interp_state = interp_state)
end

function get_pressure!(sim; duration=1,step=0.1,verbose=true, R=CartesianIndices(sim.flow.p),
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
            # interpolated = SMatrix{2,41,Float32,82}(interpolated)
            sim.sim.body = ParametricBodies.update!(sim.sim.body, interpolated, Δt)
            # @show sim.sim.body
            sim_step!(sim,tᵢ;remeasure)

            raw    = WaterLily.total_force(sim)
            scaled = raw ./ (0.5 * sim.L * sim.U^2)

            if in_period
                periodic_force += scaled[1]
                if t - t_start >= period
                    in_period = false
                end
                # return periodic_force
            end
            t += Δt
        end
        p = Array(sim.flow.p)  # Copy pressure to CPU
        pmax = maximum(abs, p)

        @show extrema(p), typeof(p), size(p)
        @assert all(isfinite, p) "Pressure contains NaNs or Infs!"  
        # Fallback in case of weird pressure values
        if isnothing(pmax) || !isfinite(pmax) || pmax == 0
            pmax = 1e-5  # or any small fixed number
        end

        clim = (-50, 50)  # Or choose a fixed range for consistency
        # @show WaterLily.div(I,sim.flow.u)
        # heatmap(p'; clims=clim, colorbar=true, title="Pressure", axis=nothing, kv...)
        
        @inside sim.flow.σ[I] = WaterLily.div(I,sim.flow.u)
        max_div = maximum(abs, sim.flow.σ[R] |> Array)
        @show max_div

        flood(sim.flow.σ[R]|>Array; clims=(-0.3,0.3), kv...)
        plotbody && body_plot!(sim)

        verbose && println("t=",round(t,digits=4),
                    ", Δt=",round(sim.flow.Δt[end],digits=3))
    end
end


#OLD GET_Forces!
# function get_forces!(sim, tᵢ) Tp = eltype(sim.flow.p) v = Float32(0); s = Float32(0) # current solver time (use whatever you consider the ground truth; or track it yourself) t = sum(sim.flow.Δt) # note: includes last step; adjust if you meant [1:end-1] ts = Tp[] # time history dts = Tp[] # Δt history f_hist = Tp[] # (Fx,Fy) per step, optional period = Tp(3) * sim.L / sim.U in_period = true periodic_force = zero(Tp) t_start = 0 last_force = nothing t_target = Tp(tᵢ) * sim.L / sim.U while t < t_target # Δt = sim.flow.Δt[end] sim.flow.Δt[end] = WaterLily.CFL(sim.flow; Δt_max=Tp(0.1)) Δt = sim.flow.Δt[end] @show Δt push!(ts, t) push!(dts, Δt) # update geometry interpolated, v, s = interpolate_cps_hermite(new_cps_list, t, Δt, sim, v, s, periodic_force) interpolated = SMatrix{2,41,Float32,82}(interpolated) sim.sim.body = ParametricBodies.update!(sim.sim.body, interpolated, Δt) # advance one step (do NOT pass tᵢ each iteration; step once) sim_step!(sim; remeasure=true) raw = WaterLily.total_force(sim) scaled = raw ./ (0.5 * sim.L * sim.U^2) last_force = scaled push!(f_hist, scaled[1]) if in_period periodic_force += scaled[1] if t - t_start >= period in_period = false end # return periodic_force end t += Δt end if last_force === nothing raw = WaterLily.total_force(sim) last_force = raw ./ (0.5 * sim.L * sim.U^2) end # Return histories (and the last force if you need it) return (force_last = last_force, ts = ts, dts = dts, forces = f_hist, periodic_force = periodic_force) end


function avg_cps_change(cps_list)
    avg_changes = Float32[]
    n = length(cps_list)
    for i in 2:n
        change = norm(cps_list[i] - cps_list[i-1]) / length(cps_list[i])
        push!(avg_changes, change)
    end
    plt = plot(1:n-1, avg_changes; xlabel="Interval", ylabel="Avg. CPS Change", title="Average CPS Change Between Intervals", legend=false)
    return plt
end

# dτ_local = Tp[]
# n = length(τ_locals)
# for i in 1:n
#     if i === 1
#         push!(dτ_local, τ_locals[1])
#         continue
#     end
#     dτ_local_val = τ_locals[i] - τ_locals[i-1]
#     push!(dτ_local, dτ_local_val)
# end