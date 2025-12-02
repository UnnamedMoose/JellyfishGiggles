using LinearAlgebra, Printf, Statistics, Plots, ParametricBodies, StaticArrays
using WaterLily, CUDA
using GeometryBasics, Optim
using BiotSavartBCs
using Interpolations: Flat, LinearInterpolation
using DelimitedFiles, DataFrames
using CairoMakie
using Dierckx

T = Float64
function cps_optimizer()
    cps_0 = SA{T}[0.000  0.024  0.064  0.175  0.323  0.481  0.643  0.798  0.996  1.191  1.228  1.201  1.171  1.178  1.154  0.804  0.606  0.475  0.397  0.340  0.323  
                    0.000  0.193  0.333  0.475  0.578  0.623  0.627  0.614  0.571  0.524  0.484  0.213  0.216  0.412  0.456  0.451  0.420  0.348  0.266  0.151 0.000  ] 
    cps_1 = SA{T}[0.000  0.024  0.067  0.175  0.326  0.488  0.639  0.794  0.993  1.164  1.198  1.296  1.265  1.171  1.090  0.801  0.606  0.478  0.404  0.343  0.323  
                    0.000  0.193  0.326  0.469  0.564  0.609  0.606  0.590  0.537  0.501  0.470  0.216  0.206  0.412  0.426  0.438  0.396  0.335  0.256  0.154  0.000  ] 
    cps_2 = SA{T}[0.000  0.024  0.081  0.188  0.333  0.481  0.639  0.798  0.986  1.150  1.222  1.390  1.373  1.205  1.154  0.801  0.616  0.501  0.427  0.370  0.323  
                    0.000  0.193  0.319  0.455  0.547  0.589  0.583  0.550  0.473  0.400  0.350  0.173  0.159  0.304  0.334  0.380  0.366  0.311  0.243  0.148 0.000  ] 
    cps_3 = SA{T}[0.000  0.034  0.091  0.199  0.337  0.481  0.629  0.781  0.973  1.077  1.181  1.346  1.319  1.198  1.154  0.798  0.643  0.522  0.448  0.387  0.357  
                    0.000  0.193  0.322  0.448  0.541  0.568  0.566  0.533  0.449  0.412  0.392  0.149  0.135  0.291  0.301  0.343  0.342  0.287  0.226  0.141  0.000  ]    
    cps_4 = SA{T}[0.000  0.027  0.081  0.199  0.357  0.478  0.626  0.781  0.976  1.178  1.222  1.228  1.191  1.195  1.151  0.801  0.633  0.525  0.438  0.384  0.347  
                    0.000  0.204  0.326  0.466  0.562  0.589  0.587  0.560  0.507  0.467  0.437  0.200  0.206  0.359  0.389  0.381  0.363  0.315  0.233  0.158  0.000  ] 
    cps_5 = SA{T}[0.000  0.027  0.074  0.182  0.340  0.471  0.629  0.791  0.989  1.185  1.222  1.151  1.124  1.164  1.127  0.798  0.616  0.498  0.407  0.360  0.337  
                    0.000  0.204  0.340  0.483  0.585  0.620  0.624  0.608  0.561  0.518  0.478  0.237  0.250  0.420  0.454  0.452  0.410  0.352  0.253  0.172  0.000  ] 
    cps_6 = SA{T}[0.000  0.027  0.074  0.182  0.330  0.464  0.629  0.794  0.989  1.185  1.225  1.158  1.124  1.164  1.141  0.794  0.613  0.488  0.394  0.343  0.330  
                    0.000  0.204  0.340  0.483  0.592  0.637  0.641  0.628  0.592  0.532  0.495  0.233  0.250  0.403  0.454  0.469  0.424  0.366  0.267  0.175  0.000  ] 
    cps_7 = SA{T}[0.000  0.027  0.074  0.182  0.330  0.464  0.629  0.794  0.989  1.185  1.225  1.178  1.144  1.164  1.141  0.794  0.613  0.488  0.394  0.343  0.330  
                    0.000  0.204  0.340  0.483  0.592  0.637  0.641  0.628  0.592  0.532  0.495  0.227  0.230  0.403  0.454  0.469  0.424  0.366  0.267  0.175  0.000  ] 
    cps_8 = SA{T}[0.000  0.027  0.074  0.182  0.330  0.464  0.629  0.794  0.989  1.185  1.225  1.185  1.151  1.164  1.141  0.794  0.613  0.488  0.394  0.343  0.330  
                    0.000  0.204  0.340  0.483  0.592  0.637  0.641  0.628  0.592  0.532  0.495  0.230  0.233  0.403  0.454  0.469  0.424  0.366  0.267  0.175  0.000  ] 
    cps_9 = SA{T}[0.000  0.027  0.074  0.182  0.330  0.464  0.629  0.794  0.989  1.185  1.225  1.191  1.154  1.164  1.141  0.794  0.613  0.488  0.394  0.343  0.330  
                    0.000  0.204  0.340  0.483  0.592  0.637  0.641  0.628  0.592  0.532  0.495  0.220  0.220  0.403  0.454  0.469  0.424  0.366  0.267  0.175  0.000  ] 


    # Linear resampling helper along arc-length for a single 2D trajectory
    function resample_constant_speed(points::Vector{SVector{2,Float64}}, K::Int)
        N = length(points)
        # cumulative arc length
        s = zeros(Float64, N)
        for i in 2:N
            s[i] = s[i-1] + norm(points[i] - points[i-1])
        end
        L = s[end]
        if L == 0.0
            # No motion: just repeat the same point
            return [points[1] for _ in 1:K]
        end
        # target uniform arc lengths
        starget = range(0.0, L, length=K)
        out = Vector{SVector{2,Float64}}(undef, K)
        j = 1
        for (k, sk) in enumerate(starget)
            # advance segment index so that s[j] <= sk <= s[j+1]
            while j < N && s[j+1] < sk
                j += 1
            end
            if sk ≤ s[1]
                out[k] = points[1]
            elseif sk ≥ s[end]
                out[k] = points[end]
            else
                # interpolate within segment [j, j+1]
                t = (sk - s[j]) / (s[j+1] - s[j] + eps())
                out[k] = (1 - t) * points[j] + t * points[j+1]
            end
        end
        return out
    end

    function resample_by_arclength(curve, N::Int)
        s_vals = range(0, 1; length=500)  
        pts = [curve(s) for s in s_vals]

        dists = cumsum([0.0; [norm(pts[i+1] - pts[i]) for i in 1:length(pts)-1]])
        total_length = dists[end]
        arc_positions = range(0, stop=total_length, length=N)
        itp = LinearInterpolation(dists, s_vals, extrapolation_bc=Flat())
        s_resampled = [itp(l) for l in arc_positions]
        points = [curve(s) for s in s_resampled]

        return SMatrix{2,N,Float64}(reduce(hcat, points))
    end

    function optimize_control_points(cps::SMatrix{2,N,T}, reference_area;
                                    λ_area::T = T(1e-1),
                                    λ_shape::T = T(1e-3),
                                    degree::Int = 2,
                                    nsamples::Int = 500) where {N,T}

        fixed_pt = cps[:, 1:3]  
        last_pt = cps[:, 11:end]
        fixed_pts = cps[:, ]

        x0_inner =  vec(Matrix(cps[:, 4:10]))
        n_inner = length(x0_inner) ÷ 2
        s_vals = range(0, stop=1, length=nsamples)

        cost = function (x::AbstractVector)
            X_inner = reshape(x, 2, n_inner)
            
            X_full = hcat(fixed_pt, X_inner, last_pt)
            cps_new = SMatrix{2,N,T}(Tuple(vec(X_full))...)

            curve = BSplineCurve(cps_new; degree=degree)
            pts = [curve(s) for s in s_vals]

            area = poly_area(pts)
            area_error = area - reference_area
            shape_error = sum(abs2, x .- x0_inner)

            return (λ_area * area_error^2) + (λ_shape * shape_error)
        end
        res = optimize(cost, copy(x0_inner), NelderMead())
        xopt_inner = Optim.minimizer(res)
        Xopt_inner = reshape(xopt_inner, 2, n_inner)

        Xopt_full = hcat(fixed_pt, Xopt_inner, last_pt)
        return SMatrix{2,N,T}(Tuple(vec(Xopt_full))...)
    end           

    function make_symmetric_jelly_new(cps_list::AbstractVector{<:SMatrix{2,N,T}}; tol = nothing) where {N,T}
    tol === nothing && (tol = sqrt(eps(T)))

    first_cps = cps_list[1]
    y0_first  = first_cps[2, 1] 

    keep_idxs = findall(j -> abs(first_cps[2, j] - y0_first) > tol, 1:N)
    K = length(keep_idxs)
    M = N + K
    keep_rev = Tuple(reverse(keep_idxs))

    build_sym_jelly(cps::SMatrix{2,N,T}) where {N,T} = SMatrix{2,M,T}(
            ntuple(2M) do k
                if k <= 2N
                    i = (k - 1) % 2 + 1
                    j = (k + 1) ÷ 2
                    cps[i, j]
                elseif k <= 2N + 2K
                    y0  = cps[2, 1]

                    kk  = k - 2N
                    i   = (kk - 1) % 2 + 1
                    jth = (kk + 1) ÷ 2
                    col = keep_rev[jth]

                    if i == 1
                        cps[1, col]           
                    else
                        2*y0 - cps[2, col]     
                    end
                else
                    0
                end
            end
        )

        map(build_sym_jelly, cps_list)
    end

    reverse_cps_order(cps::SMatrix{2,N,T}) where {N,T} =       
        SMatrix{2,N,T}(cps[:, reverse(1:N)])                       

    reverse_cps_list(cps_list::AbstractVector{<:SMatrix{2,N,T}}) where {N,T} =
        map(reverse_cps_order, cps_list)

    start = SA{T}[0.000 0.000 0.000 0.000;
                0.000 -0.010 -0.020 -0.030]
    ending = SA{T}[0.000 0.000 0.000 0.000;
                0.030 0.020 0.010 0.000]
    to_static(cps::AbstractMatrix{T}) where {T} = SMatrix{size(cps,1), size(cps,2), T}(cps)
    # --- Main block ---
    cps_list_ini = [cps_0, cps_1, cps_2, cps_3, cps_4, cps_5, cps_6, cps_7, cps_8, cps_0]
    Npoints = 50

    cps_list_og = [
        begin
            mid = to_static(cps[:,1:end])                     # convert slice → SMatrix
            curve = BSplineCurve(mid; degree=2)              # your own BSplineCurve
            mid_resampled = resample_by_arclength(curve, Npoints)
            # hcat(cps[:,1:6], mid_resampled, cps[:,18:end])   # recombine
            mid_resampled
        end
        for cps in cps_list_ini
    ]

    # cps_list_ini     = [cps_0, cps_1, cps_2, cps_3, cps_4, cps_5, cps_6, cps_7, cps_8, cps_9]
    # curve_0         = BSplineCurve(cps_0[:,8:17])
    # curves          = [BSplineCurve(SA{T}(cps[:,8:17]); degree=2) for cps in cps_list_ini]
    # # Npoints         = 25
    # # cps_list_p2     = [resample_by_arclength(BSplineCurve(cps[:,8:17]; degree=2), Npoints) for curve in curves]
    # cps_list_og     = [hcat(cps[1:7], resample_by_arclength(BSplineCurve(cps[:,8:17]; degree=2), Npoints), cps[18:end]) for cps in cps_list_ini]

    # Pack frames into a vector for convenience
    # cps = SMatrix{2,21,Float64}[
    #     cps_0, cps_1, cps_2, cps_3, cps_4,
    #     cps_5, cps_6, cps_7, cps_8, cps_9
    # ]
    cps = Vector{SMatrix{2,50,Float64,100}}(cps_list_og)
    K = length(cps)             # number of time steps (10)
    M = size(cps[1], 2)         # number of control points (21)

    # Build new per-frame control sets with constant-speed columns
    cps_const = [similar(cps[1]) for _ in 1:K]  # vector of 2×21 matrices
    trajectories = []
    for col in 1:M
        # trajectory of this control point across frames
        traj = [cps[k][:, col] for k in 1:K]  # Vector of SVector{2}
        traj_resampled = resample_constant_speed(traj, K)
        push!(trajectories, traj_resampled)
        # write back into per-frame matrices
        for k in 1:K
            cps_const[k][:, col] = traj_resampled[k]
        end
    end

    cps_const = Vector{SMatrix{2,50,Float64,100}}(cps_const)

    cps_list        = make_symmetric_jelly_new(cps_const)     
    cps_list        = [hcat(cps, cps[:, 1]) for cps in cps_list]                 

    new_cps_list    = reverse_cps_list(cps_list)                 
    # new_cps_list    = [hcat(start .- shift[:,i], cps[:,2:end-1], ending .- shift[:,i]) for (i,cps) in enumerate(new_cps_list)]
    new_cps_list    = [hcat(start, cps[:,2:end-1], ending) for (i,cps) in enumerate(new_cps_list)]
    new_cps_list    = [SMatrix{2, 105, Float64, 210}(cps) for cps in new_cps_list]        

    function smooth_spline_variable_upsample(cps_seq; s=0.001, up1=10, up2=5, split=nothing)
        Nframes = length(cps_seq)
        @show Nframes
        Npts    = size(cps_seq[1], 2)
        t       = 1:Nframes
        mid     = split === nothing ? (Nframes ÷ 2) : split
        @assert 1 ≤ mid < Nframes "split must be in 1:(Nframes-1)"

        # Build two interpolation grids (avoid duplicate at the junction)
        t1 = range(first(t), mid; length = (mid - first(t)) * up1 + 1)
        t2 = range(mid, last(t);  length = (last(t) - mid) * up2 + 1)
        @show  (mid - first(t)) * up1 + 1
        @show  (last(t) - mid) * up2 + 1
        t_interp = [t1; t2[2:end]]  # drop duplicate 'mid'

        # Fit splines per control point (once)
        splx = Vector{Spline1D}(undef, Npts)
        sply = Vector{Spline1D}(undef, Npts)
        for i in 1:Npts
            xs = [cps_seq[k][1,i] for k in t]
            ys = [cps_seq[k][2,i] for k in t]
            splx[i] = Spline1D(t, xs; k=3, s=s)
            sply[i] = Spline1D(t, ys; k=3, s=s)
        end

        # Evaluate at nonuniform times
        out = Vector{SMatrix{2, Npts, Float64}}(undef, length(t_interp))
        for (j, τ) in enumerate(t_interp)
            M = Matrix{Float64}(undef, 2, Npts)
            for i in 1:Npts
                M[1,i] = splx[i](τ)
                M[2,i] = sply[i](τ)
            end
            out[j] = SMatrix{2, Npts, Float64, 210}(M)
        end
        return out, t_interp
    end

    dense_cps, interp_cps = smooth_spline_variable_upsample(new_cps_list; s=0.01, up1=10, up2=10)
    @show typeof(dense_cps)
    
    return dense_cps
end

control_points_list = cps_optimizer()
n_cps = length(control_points_list[1][1,:])
cps_paths_x = [[] for _ in 1:n_cps]  # vector of vectors
cps_paths_y = [[] for _ in 1:n_cps]
for j in 1:91
    for (i, p) in enumerate(control_points_list[j][1,:])
        push!(cps_paths_x[i], p)
    end

    for (i, p) in enumerate(control_points_list[j][2,:])
        push!(cps_paths_y[i], p)
    end
end

function exp_smooth(x::Vector{T}, α::T) where {T<:Real}
    s = similar(x)
    s[1] = x[1]
    for t in 2:length(x)
        s[t] = α * x[t] + (1 - α) * s[t-1]
    end
    return s
end


@inline function exp_interp(f₀::AbstractArray, f₁::AbstractArray, t, t₀, t₁, τ)
    γ = (1-expm1(-(t-t₀)/τ)) / (1-expm1(-(t₁ - t₀)/τ))
    return f₀ + (f₁ - f₀) * γ
end

τ = 3
t = 0.45
steps = length(control_points_list)
times = range(0, τ; length=steps)
k = searchsortedlast(times, t)
k = clamp(k, 1, length(times)-1)
t₀, t₁ = times[k], times[k+1]
f₀, f₁ = control_points_list[k], control_points_list[k+1]
G_interp = exp_interp(f₀, f₁, t, t₀, t₁, τ)

τ = 3
Δt = 0.1
α = 0.2 #-expm1(-Δt/τ)
N = length(control_points_list[1][1,:])
sx = Vector{Vector{T}}(undef, N)
sy = Vector{Vector{T}}(undef, N)
cps = Vector{Matrix{T}}(undef, N)

for i in 1:N
    sx[i] = exp_smooth(T.(cps_paths_x[i]), α)
    sy[i] = exp_smooth(T.(cps_paths_y[i]), α)
    cps[i] = hcat(sx[i], sy[i])'
end

# Plots.plot(cps_paths_x[55], cps_paths_y[55], color=:red)
# Plots.plot!(cps[55][1,:], cps[55][2,:], color=:blue)

# cps = control_points_list
# K = length(cps)             # number of time steps (10)
# M = size(cps[1], 2)         # number of control points (21)

# cps_const = [similar(cps[1]) for _ in 1:K]  # vector of 2×21 matrices
# trajectories = []
# for col in 1:M
#     traj = [cps[k][:, col] for k in 1:K]  # Vector of SVector{2}
# end

# Plot: each control point gets its own velocity curve over time
# plt = Plots.plot(title="Geometry Shapes")
# set_times = [1,10,20,30,40,50,60,70,80,90]
# for j in set_times
#     Plots.scatter!(control_points_list[j][1,:], control_points_list[j][2,:], label="Shape $j")
# end
# display(plt)

# plt = Plots.plot(title="Geometry Shapes", legend=:false)
# for j in 1:10
#     Plots.plot!(cps_list_ini[j][1,:], cps_list_ini[j][2,:], label="Shape $j")
# end
# display(plt)

# # Convert to per-control-point arrays
# function control_point_trajectories(cps_list::Vector{<:SMatrix})
#     nframes = length(cps_list)
#     N = size(cps_list[1], 2)
    
#     x_traj = [ [ cps_list[t][1, j] for t in 1:nframes ] for j in 1:N ]
#     y_traj = [ [ cps_list[t][2, j] for t in 1:nframes ] for j in 1:N ]
    
#     return x_traj, y_traj
# end

# x_traj, y_traj = control_point_trajectories(control_points_list)

# for i in 1:length(x_traj)
#     x_traj[i] = vcat(x_traj[i], x_traj[i][1])
# end

# cps_0 = SA{T}[0.000  0.024  0.064  0.175  0.323  0.481  0.643  0.798  0.996  1.191  1.228  1.201  1.171+sin(t)  1.178+sin(t)  1.154  0.804  0.606  0.475  0.397  0.340  0.323  
#                 0.000  0.193  0.333  0.475  0.578  0.623  0.627  0.614  0.571  0.524  0.484  0.213  0.216  0.412  0.456  0.451  0.420  0.348  0.266  0.151 0.000  ] 
 
# display(Plots.scatter(cps_0[1,:], cps_0[2,:]))





















# function sinusoidal_cps(base_cps::SMatrix{2,N,T}, t; A_scale=0.1, freq=1.0, phase=0.0) where {N,T}
#     x = base_cps[1, :]
#     y0 = base_cps[2, :]
    
#     # Amplitude profile: larger near margin, smaller near apex
#     s = range(0, 1; length=N)
#     A = A_scale .* (sin.(π .* s)).^2  # smooth amplitude shape

#     y = y0 .+ A .* sin.(2π * freq * t .+ phase)
#     return SMatrix{2,N,T}(vcat(x', y'))
# end

# function get_cps_list(base_cps, period, nframes)
#     times = range(0, period; length=nframes)
#     [sinusoidal_cps(base_cps, t; A_scale=0.1, freq=1/period) for t in times]
# end

# === Example Data (your control point) ===
# x_data = [0.000  0.024  0.064  0.175  0.323  0.481  0.643  0.798  0.996  1.191  1.228  1.201  1.171  1.178  1.154  0.804  0.606  0.475  0.397  0.340  0.323] 
# y_data = [0.000  0.193  0.333  0.475  0.578  0.623  0.627  0.614  0.571  0.524  0.484  0.213  0.216  0.412  0.456  0.451  0.420  0.348  0.266  0.151 0.000]
# Δt = 1.0                             # assume 1 time unit between samples
# t_data = collect(0:Δt:(length(x_data)-1)*Δt)

# """
#     fit_sine_fixedP(t, y; P)

# Fit y(t) ≈ a0 + A*sin(2π*t/P + φ)
# Returns (a0, A, φ, ω, P)
# """
# function fit_sine_fixedP(t, y; P)
#     ω = 2π / P
#     Φ = [ones(length(t))  sin.(ω .* t)  cos.(ω .* t)]
#     θ = Φ \ y
#     a0, a1, b1 = θ
#     A = hypot(a1, b1)
#     φ = atan(-b1, a1)
#     (; a0, A, φ, ω, P)
# end

# ysine(t, p) = p.a0 .+ p.A .* sin.(p.ω .* t .+ p.φ)

# function fit_all_controlpoints(x_traj, y_traj; P, t=nothing)
#     npoints = length(x_traj)
#     nframes = length(x_traj[1])
#     if t === nothing
#         t = collect(0:nframes-1)
#     end

#     fitx = Vector{Any}(undef, npoints)
#     fity = Vector{Any}(undef, npoints)

#     for j in 1:npoints
#         fitx[j] = fit_sine_fixedP(t, x_traj[j]; P=P)
#         fity[j] = fit_sine_fixedP(t, y_traj[j]; P=P)
#     end
#     return fitx, fity
# end

# P = 10.0  # stroke period
# fitx, fity = fit_all_controlpoints(x_traj, y_traj; P)

# function plot_controlpoint_fit(j, x_traj, y_traj, fitx, fity; t=collect(0:length(x_traj[1])-1))
#     ts = range(first(t), last(t), length=200)
#     xfit = ysine(ts, fitx[j])
#     yfit = ysine(ts, fity[j])

#     fig = Figure(resolution=(900,300))
#     ax1 = Axis(fig[1,1], title="Control point $j: x(t)", xlabel="time", ylabel="x")
#     CairoMakie.scatter!(ax1, t, x_traj[j], color=:black)
#     lines!(ax1, ts, xfit, color=:blue)

#     ax2 = Axis(fig[1,2], title="y(t)", xlabel="time", ylabel="y")
#     CairoMakie.scatter!(ax2, t, y_traj[j], color=:black)
#     lines!(ax2, ts, yfit, color=:red)
#     display(fig)
# end

# plot_controlpoint_fit(10, x_traj, y_traj, fitx, fity)

# function make_symmetric_jelly_new(cps_list::AbstractVector{<:SMatrix{2,N,T}}; tol = nothing) where {N,T}
#     tol === nothing && (tol = sqrt(eps(T)))

#     first_cps = cps_list[1]
#     y0_first  = first_cps[2, 1] 

#     keep_idxs = findall(j -> abs(first_cps[2, j] - y0_first) > tol, 1:N)
#     K = length(keep_idxs)
#     M = N + K
#     keep_rev = Tuple(reverse(keep_idxs))

#     build_sym_jelly(cps::SMatrix{2,N,T}) where {N,T} = SMatrix{2,M,T}(
#         ntuple(2M) do k
#             if k <= 2N
#                 i = (k - 1) % 2 + 1
#                 j = (k + 1) ÷ 2
#                 cps[i, j]
#             elseif k <= 2N + 2K
#                 y0  = cps[2, 1]

#                 kk  = k - 2N
#                 i   = (kk - 1) % 2 + 1
#                 jth = (kk + 1) ÷ 2
#                 col = keep_rev[jth]

#                 if i == 1
#                     cps[1, col]           
#                 else
#                     2*y0 - cps[2, col]     
#                 end
#             else
#                 0
#             end
#         end
#     )

#     map(build_sym_jelly, cps_list)
# end

# reverse_cps_order(cps::SMatrix{2,N,T}) where {N,T} =       
# SMatrix{2,N,T}(cps[:, reverse(1:N)])                       

# reverse_cps_list(cps_list::AbstractVector{<:SMatrix{2,N,T}}) where {N,T} =
#     map(reverse_cps_order, cps_list)

# # start = SA{T}[0.000 0.000 0.000 0.000;
# #             0.000 -0.010 -0.020 -0.030]
# # ending = SA{T}[0.000 0.000 0.000 0.000;
# #             0.030 0.020 0.010 0.000]
# # cps_list        = make_symmetric_jelly_new(cps_list_og)     
# # cps_list        = [hcat(cps, cps[:, 1]) for cps in cps_list]                 

# # new_cps_list    = reverse_cps_list(cps_list)                 
# # # new_cps_list    = [hcat(start .- shift[:,i], cps[:,2:end-1], ending .- shift[:,i]) for (i,cps) in enumerate(new_cps_list)]
# # new_cps_list    = [hcat(start, cps[:,2:end-1], ending) for (i,cps) in enumerate(new_cps_list)]
# # new_cps_list    = [SMatrix{2, 55, Float64, 110}(cps) for cps in new_cps_list]        



# function get_cps(time)
#     xfit = zeros(1,21)
#     yfit = zeros(1,21)
#     for i in 1:21
#         xfit[i] = ysine(time, fitx[i])
#         yfit[i] = ysine(time, fity[i])
#     end
#     cpsfit = vcat(xfit, yfit)
#     return cpsfit
# end

# cps_list = [SMatrix{2,21,Float64,42}(get_cps(t)) for t in 1:10]
# cps_list_sym = make_symmetric_jelly_new(cps_list)
# cps_list_sym = [hcat(cps, cps[:, 1]) for cps in cps_list_sym]
# cps_list_sym = reverse_cps_list(cps_list)
# cps_list_sym = [hcat(start, cps[:,2:end-1], ending) for (i,cps) in enumerate(new_cps_list)]

# plt = Plots.plot()
# for i in 1:10
#     Plots.plot!(cps_list_sym[i][1,:], cps_list_sym[i][2,:])
# end
# display(plt)

# plt2 = Plots.plot()
# for i in 1:10
#     Plots.plot!(new_cps_list[i][1,:], new_cps_list[i][2,:])
# end
# display(plt2)

# # === 2. Generate smooth fitted curve ===
# t_smooth = range(0, stop=t_data[end], length=200)
# x_fit = ysine(t_smooth, fitx)
# y_fit = ysine(t_smooth, fity)

# # === 3. Plot results ===
# fig = Figure(resolution=(800,400))

# # (a) X(t)
# ax1 = Axis(fig[1,1], title="X-coordinate motion", xlabel="time", ylabel="x")
# CairoMakie.scatter!(ax1, t_data, x_data, color=:black, label="data")
# lines!(ax1, t_smooth, x_fit, color=:blue, linewidth=2, label="fit")
# axislegend(ax1, position=:rb)

# # (b) Y(t)
# ax2 = Axis(fig[1,2], title="Y-coordinate motion", xlabel="time", ylabel="y")
# CairoMakie.scatter!(ax2, t_data, y_data, color=:black, label="data")
# lines!(ax2, t_smooth, y_fit, color=:red, linewidth=2, label="fit")
# axislegend(ax2, position=:rb)

# display(fig)
