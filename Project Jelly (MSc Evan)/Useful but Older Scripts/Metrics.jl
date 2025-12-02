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

function get_curves(new_cps_list)
    colors = [:red, :blue, :green, :orange, :purple, :yellow, :brown, :pink, :gray, :cyan]
    plt = Plots.plot(title="NURBS Curves")
    for (i,cps) in enumerate(new_cps_list)
        T = Float32
        n_ctrl = size(cps, 2)
        weights = ones(T, n_ctrl)
        degree = 2
        knots = T.(clamped_uniform_knots(degree, n_ctrl))
        int_curve = NurbsCurve(cps, knots, weights)
        x, y = cps[1, :], cps[2, :]
        Plots.plot!(plt, int_curve, 
        color=colors[i], 
        fillalpha=0.2, 
        alpha = 0.5,
        add_cp=false, 
        label="Curve $i", 
        legend=:topright)
        # println("Self-intersection? ", self_intersects(x,y))
    end
    for (i,cps) in enumerate(new_cps_list)
        Plots.scatter!(cps[1,:], cps[2,:],color=colors[i])
    end
    display(plt)
end

function plot_interp_shapes(new_cps_list, period=3, start=0, duration=3, step=0.1)
    interp_set = []
    intersect_state = []
    t_test = start:step:duration
    for t in t_test
        interp, k, τ_total, state = interpolate_cps_hermite_new(new_cps_list, t, period)
        push!(interp_set, interp)
        push!(intersect_state, state)
    end
    get_curves(interp_set)
end

function get_shape_error(new_cps_list)
    s_vals          = range(0, 1; length=100)
    crvs            = [BSplineCurve(cps; degree=2) for cps in new_cps_list]
    points          = [[curve(s) for s in s_vals] for curve in crvs]                                        # Evaluate each curve at the sampled points
    areas           = [poly_area(pts) for pts in points]                                                    # Calculate the area of each polygon defined by the points
    rel_area        = [area / areas[1] for area in areas]                                                   # Relative area compared to the first shape        
    rel_errors      = [(area - areas[1]) / areas[1] * 100 for area in areas]    

    println("Shape Area Summary:")
    println("────────────────────────────────────────────")
    @printf("%-10s %-15s %-15s\n", "Curve t/T", "Area", "Δ% from t = 0")
    println("────────────────────────────────────────────")
    n = length(areas)
    for i in 1:n
        @printf("Curve %-4d  %-15.6f %-15.2f\n", i, areas[i], rel_errors[i])
    end

    # plot(1:length(areas), areas; label="Area", xlabel="t/T", ylabel="Area", linewidth=2, legend=:topleft, title="Area per time step")
    plt = Plots.plot(1:length(rel_area), rel_area; label="Relative Area [-]", xlabel="t/T [-]", ylabel="Relative Area [-]",  linewidth=2, legend=:false, linestyle=:dash, title="Relative Area per time step")
    display(plt)
    savefig(plt, "relative_area.png")
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
    s_vals          = range(0, 1; length=100)            
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

## Function to generate the static signed distance function plots of each control point set.
function generate_sdf_plots(new_cps_list, thk=2.0, D=2^7, Tp=Float32, degree=3)
    save_dir = joinpath(pwd(), "SDF_plots")
    isdir(save_dir) || mkpath(save_dir)
    for (i, cps) in enumerate(new_cps_list)
        Tp = Float32
        n_ctrl = size(cps, 2)
        weights = ones(T, n_ctrl)
        knots = T.(clamped_uniform_knots(degree, n_ctrl))
        curve = NurbsCurve(cps .* 2D .+ SA{Tp}[D,3*D], knots, weights)         
        body = DynamicNurbsBody(curve; thk=thk, boundary=true)

        xs = range(0, 6.25 * D, length=200)
        ys = range(0, 6.25 * D, length=200)
        Z = [sdf(body, SA[x, y]) for y in ys, x in xs]

        signed_df = Plots.heatmap(xs, ys, Z; color=:viridis, aspect_ratio=1, title="Signed Distance Field $i")
        Plots.contour!(xs, ys, Z, levels=[0.0], linewidth=2, color=:red)  # Contour where sdf=0
        # plot!(body.curve, shift=(0.5, 0.5), alpha=0.8, add_cp=true)
        display(signed_df)
        savefig(signed_df, joinpath(save_dir, "sdf_nurbs_$(i).png"))
    end
end

function generate_grid_view(new_cps_list, D=2^5)
    plot_cps_list   = new_cps_list .* 2 .* D 
    crvs            = [BSplineCurve(cps; degree=3) for cps in plot_cps_list]  
    # Plot grid cells behind jelly foer different sizes.
    colors = [:red, :blue, :green, :orange, :purple, :yellow]
    yrange          = -2D:1:2D #-D/2:1:D/2
    xrange          = 0:1:4D #0:1:1.25*D
    grid_view             = plot(aspect_ratio=1, xlabel="x", ylabel="y", xlims=(1.75D,2.5D), ylims=(-1.5D,0),
            title="Grid Size", legend=false)
            
    for x in xrange
        plot!([x, x], [first(yrange), last(yrange)],
            color=:gray, alpha=0.5, linewidth=0.5)
    end
    for y in yrange
        plot!([first(xrange), last(xrange)], [y, y],
            color=:gray, alpha=0.5, linewidth=0.5)
    end
    plot!(crvs[4], add_cp = false)
    plot!(crvs[5], color=:red, add_cp = false)
    # for curve in curves[3:4]
    #     plot!(curve,color=:red, alpha=0.5, add_cp = false)
    # end
    # for (i, cps) in enumerate(plot_cps_list[3:4])
    #     scatter!(cps[1,:], cps[2,:], color=colors[i])
    # end

    display(grid_view)
    savefig("grid_size.png")
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

# function twiny(sp::Plots.Subplot)
#     sp[:top_margin] = max(sp[:top_margin], 30Plots.px)
#     plot!(sp.plt, inset = (sp[:subplot_index], bbox(0,0,1,1)))
#     twinsp = sp.plt.subplots[end]
#     twinsp[:xaxis][:mirror] = true
#     twinsp[:background_color_inside] = RGBA{Float64}(0,0,0,0)
#     Plots.link_axes!(sp[:yaxis], twinsp[:yaxis])
#     twinsp
# end
# twiny(plt::Plots.Plot = current()) = twiny(plt[1])

using LinearAlgebra
using Statistics
using Plots

# === Helper functions === #

"""
    check_geometry(control_points; name="Curve", plot=true)

Takes a 2×N matrix of control points (rows = [x; y]) and prints diagnostics:
- Monotonicity in x
- Local curvature values and anomalies
- Plots the curve and curvature if `plot=true`
"""
function check_geometry(control_points; name="Curve", doplot=true)
    x = control_points[1, :]
    y = control_points[2, :]
    N = length(x)

    dx = diff(x)
    nonmono_idx = findall(<=(0), dx)
    if !isempty(nonmono_idx)
        println("⚠️ Non-monotonic x detected in $name at indices: ", nonmono_idx)
        println("   (x decreases or flattens locally)")
    else
        println("✅ x is strictly monotonic in $name.")
    end

    dx_dt = diff(x)
    dy_dt = diff(y)
    d2x_dt2 = diff(dx_dt)
    d2y_dt2 = diff(dy_dt)

    n = length(d2x_dt2)
    curvature = abs.(
        @. dx_dt[1:n]*d2y_dt2 - dy_dt[1:n]*d2x_dt2
    ) ./ ((@. dx_dt[1:n]^2 + dy_dt[1:n]^2).^(3/2) .+ eps())

    println("Curvature stats for $name:")
    println("   mean = ", round(mean(curvature), digits=5))
    println("   max  = ", round(maximum(curvature), digits=5))
    println("   std  = ", round(std(curvature), digits=5))

    high_curv_idx = findall(x -> x > 3mean(curvature), curvature)
    if !isempty(high_curv_idx)
        println("⚠️ High-curvature zones at indices: ", high_curv_idx)
    else
        println("✅ No unusually sharp curvature regions detected.")
    end

    if doplot
        plt1 = Plots.plot(x, y, aspect_ratio=1, title="$name: Geometry", label="Curve", lw=2)
        Plots.scatter!(plt1, x, y, label="Control points", ms=3)

        plt2 = Plots.plot(1:n, curvature, title="$name: Curvature profile", label="Curvature", lw=2)
        Plots.vline!(plt2, high_curv_idx, label="High κ", c=:red, ls=:dash)

        display(Plots.plot(plt1, plt2, layout=(1, 2), size=(1000, 400)))
    end

    return curvature
end

## Chatgpt curvature routine:
# P1, P2, P3 = new_cps_list[3][:,1], new_cps_list[3][:,2], new_cps_list[3][:,3]
# Pend, Pm1, Pm2 = new_cps_list[3][:,end], new_cps_list[3][:,end-1], new_cps_list[3][:,end-2]

# S_start = P3 - 2P2 + P1
# S_new = S_start + 2Pm1 - Pend   # replaces P_end-2
# @show Pm2 - S_new
# old_cps3 = new_cps_list[3]
# new_cps_list[3] = hcat(new_cps_list[3][:,1:end-3], S_new, new_cps_list[3][:,end-1:end])
# new_cps3 = new_cps_list[3]
# @show old_cps3 - new_cps3

# === Example usage === #

# Suppose you’ve loaded your control point sets as `discretized_set` (2×75×M)
# For a single set (2×75 SMatrix), call:

# check_geometry(Matrix(discretized_set[:,:,1]), name="Set 1")

# or, if your variable is already a list of 2×N matrices:
# for (i, cp) in enumerate(discretized_set)
#     check_geometry(cp, name="Curve $i")
# end

## Chatgpt code to diagnose SDF. Define a circle around a geometry point for further investigation.
function diagnose_artifact(body::ParametricBody, t1::Float64, t2::Float64;
                           x_center=SA[300.0, 25.0], r=35.0, npts=20)

    # Create a ring of points around a known problematic location
    θs = LinRange(0, 2π, npts+1)[1:end-1]
    xs = [x_center .+ r .* SA[cos(θ), sin(θ)] for θ in θs]

    # Track u values returned by locate
    us_t1 = [body.locate(body.map(x, t1), t1) for x in xs]
    us_t2 = [body.locate(body.map(x, t2), t2) for x in xs]

    # Track distance values from sdf
    ds_t1 = [sdf(body, x, t1) for x in xs]
    ds_t2 = [sdf(body, x, t2) for x in xs]

    # Track closest point positions (on the curve)
    pts_t1 = [body.curve(u, t1) for u in us_t1]
    pts_t2 = [body.curve(u, t2) for u in us_t2]

    # Plot 1: u-values vs index
    p1 = plot(1:npts, us_t1, label="u(t1 = $t1)", legend=:top)
    plot!(p1, 1:npts, us_t2, label="u(t2 = $t2)", title="Parameter u from locate")

    # Plot 2: signed distances
    p2 = plot(1:npts, ds_t1, label="sdf(t1)", ylabel="signed distance")
    plot!(p2, 1:npts, ds_t2, label="sdf(t2)", title="Signed distance around point")

    # Plot 3: closest points on the curve
    p3 = scatter([x[1] for x in xs], [x[2] for x in xs], label="query points", ms=4)
    scatter!(p3, [p[1] for p in pts_t1], [p[2] for p in pts_t1], label="closest @t1", ms=4)
    scatter!(p3, [p[1] for p in pts_t2], [p[2] for p in pts_t2], label="closest @t2", ms=4)
    title!(p3, "Nearest curve points")
    # aspect_ratio!(p3, :equal)

    display(p1)
    display(p2)
    display(p3)
end


"""
Older data gathering codes that might be useful for later.
"""
# data = readdlm("test_psolver.log", ',', skipstart=1, String)

# # Convert to DataFrame for easier handling
# df = DataFrame(pc = data[:,1],
#                iter = parse.(Int, data[:,2]),
#                rinf = parse.(Float64, data[:,3]),
#                r2   = parse.(Float64, data[:,4]))

# # Forward-fill missing pc values (blank entries "")
# for i in 2:nrow(df)
#     if df.pc[i] == ""
#         df.pc[i] = df.pc[i-1]
#     end
# end

# # Now split predictor vs corrector
# df_pred = filter(:pc => ==("p"), df)
# df_corr = filter(:pc => ==("c"), df)

# Residuals (log scale is typical for residuals)
# Left y-axis = forces
# forceplt = plot(res.ts[1:end], res.forces[1:end];
#      label="Force",
#      xlabel="tU/L",
#      ylabel="Non-dim. Force",
#      color=:red,
#      xgrid=true,
#      gridstyle=:dash,
#      gridalpha=0.7)

# predinfplt = plot(df_pred.rinf[17:end];
#       yscale=:log10,
#       label="L∞ predictor",
#       color=:blue,
#       xlabel = "predictor step",
#       ylabel="Residual",
#       legend=:topleft)
#       plot!(twiny(), df_corr.rinf[17:end];
#       yscale=:log10,
#       label="L∞ corrector",
#       color=:red,
#       xlabel = "corrector step")

# prediterplt = plot(df_pred.iter[1:end];
#       label = "It. predictor",
#       xlabel = "predictor step",
#       ylabel = "Iterations",
#       color =:blue,
#       legend=:topleft)
#       plot!(twiny(), df_corr.iter[1:end];
#       label = "It. corrector",
#       xlabel = "corrector step",
#       color =:red)



# savefig(forceplt, "forces.png")
# savefig(predinfplt, "Linf.png")
# savefig(prediterplt, "Iterations.png")


# corrinfplt = plot(df_corr.rinf[17:end];
#       yscale=:log10,
#       label="L∞ corrector",
#       color=:red,
#       xlabel = "step",
#       ylabel = "residual")

# corriterplt = plot(df_corr.iter[1:end];
#       label="It. corrector",
#       xlabel = "time step",
#       ylabel = "Iterations",      
#       color=:red)

# get_pressure!(sim; duration = 3, step = 0.1, remeasure = true, plotbody = true)
# times = [Tp(3), Tp(3+3/5), Tp(3+6/5), Tp(3+9/5), Tp(3+12/5), Tp(3+15/5)]
# sim_frames!(sim; duration = 15, step = 0.1, remeasure=true, plotbody=true, savepath="snapshot", dpi=300)