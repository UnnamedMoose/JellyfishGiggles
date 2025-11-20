using LinearAlgebra, Printf, Statistics, Plots, ParametricBodies, StaticArrays
using WaterLily
using GeometryBasics, Optim, CUDA
using BiotSavartBCs
using DelimitedFiles, DataFrames
using Interpolations: Flat, LinearInterpolation
using LinearAlgebra, Dierckx
using CairoMakie

@info "Running with $(Threads.nthreads()) Julia threads"

Tp = Float32
T = Float32

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

function densify_cps_list(new_cps_list::Vector{SMatrix{2, N, T, 210}}, period::Float64; 
                        frames_per_segment::Int = 2) where {N, T}
    nphases = length(new_cps_list)
    total_interp_frames = frames_per_segment * (nphases - 1)

    times = range(0, stop=(nphases - 1), length=total_interp_frames + 1) .* (period / (nphases - 1))

    return [interpolate_cps_hermite_new(new_cps_list, t, period; nphases=nphases) for t in times]
end

function create_cps_list(::Type{T}) where {T<:AbstractFloat}
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

    function resample_curve(curve, N::Int, ::Type{T}=Float32) where {T}
        s_vals = range(0, 1; length=N)
        xs = Vector{T}(undef, N)
        ys = Vector{T}(undef, N)

        for (i,s) in enumerate(s_vals)
            p = curve(s)         
            xs[i] = p[1]
            ys[i] = p[2]
        end

        return SMatrix{2,N,T}(reduce(hcat, (SVector(xs[i], ys[i]) for i in 1:N)))
    end

    function resample_by_arclength(curve, N::Int)
        s_vals = range(0, 1; length=500)  # dense sampling
        pts = [curve(s) for s in s_vals]

        dists = cumsum([0.0; [norm(pts[i+1] - pts[i]) for i in 1:length(pts)-1]])
        total_length = dists[end]
        arc_positions = range(0, stop=total_length, length=N)
        itp = LinearInterpolation(dists, s_vals, extrapolation_bc=Flat())
        # Interpolate back to find corresponding `s` values
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
    y0_first  = first_cps[2, 1] # reflection axis at y = y₀ (first control point's y)

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
                        cps[1, col]              # x unchanged
                    else
                        2*y0 - cps[2, col]       # reflect y about y = y0
                    end
                else
                    0
                end
            end
        )

        map(build_sym_jelly, cps_list)
    end
    function make_symmetric_jelly(cps_list::AbstractVector{<:SMatrix{2,N,T}};
                                        tol = nothing) where {N,T}
        tol === nothing && (tol = sqrt(eps(T)))

        first_cps = cps_list[1]::SMatrix{2,N,Float32,2N}
        keep_idxs = findall(j -> abs(first_cps[2, j]) > tol, 1:N)  
        K = length(keep_idxs)
        M = N + K + 1                                           
        keep_rev = Tuple(reverse(keep_idxs))                        

        build_sym_jelly(cps::SMatrix{2,N,T}) where {N,T} = SMatrix{2,M,T}(
            ntuple(2M) do k
                if k <= 2N                                           
                    i = (k - 1) % 2 + 1
                    j = (k + 1) ÷ 2
                    cps[i, j]
                elseif k <= 2N + 2K                                
                    kk  = k - 2N
                    i   = (kk - 1) % 2 + 1
                    jth = (kk + 1) ÷ 2
                    col = keep_rev[jth]
                    i == 1 ? cps[1, col] : -cps[2, col]
                else
                    cps[1,1]                                               
                end
            end
        )

        map(build_sym_jelly, cps_list)                                    
    end

    reverse_cps_order(cps::SMatrix{2,N,T}) where {N,T} =       
        SMatrix{2,N,T}(cps[:, reverse(1:N)])                       

    reverse_cps_list(cps_list::AbstractVector{<:SMatrix{2,N,T}}) where {N,T} =
        map(reverse_cps_order, cps_list)

    # shift           = SA{T}[0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9;
    #                 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
    shift           = SA{T}[0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9;
                    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
    start = SA{T}[0.000 0.000 0.000 0.000;
            0.000 -0.010 -0.020 -0.030]
    ending = SA{T}[0.000 0.000 0.000 0.000;
                0.030 0.020 0.010 0.000]
    cps_list_og     = [cps_0, cps_1, cps_2, cps_3, cps_4, cps_5, cps_6, cps_7, cps_8, cps_9] 
    # cps_list_og     = [cps_0 .- shift[:,1], cps_0 .- shift[:,2], cps_0 .- shift[:,3], cps_0 .- shift[:,4], cps_0 .- shift[:,5], cps_0 .- shift[:,6], cps_0 .- shift[:,7], cps_0 .- shift[:,8], cps_0 .- shift[:,9], cps_0 .- shift[:,10]]
    # cps_list_og    = make_circle_cps_sin_motion(Float64, 22, 10; θ₁=π,  radius=0.25,freq=0.1, amplitude=(0.25, 0.25))
    # cps_list_og     = [cps_4, cps_5, cps_4, cps_5, cps_4, cps_5, cps_4, cps_5, cps_4, cps_5] 
    
    s_vals          = range(0, stop=1, length=500)                                                         
    ref_crv         = BSplineCurve(cps_list_og[1]; degree=2)
    ref_points      = [ref_crv(s) for s in s_vals]                                                          
    reference_area  = poly_area(ref_points)                                                               
    opt_cps_list    = [optimize_control_points(cps, reference_area) for cps in cps_list_og]
    # opt_cps_list    = cps_list_og
    curves          = [BSplineCurve(cps; degree=2) for cps in opt_cps_list]
    Npoints         = 50
    cps_list_og     = [resample_by_arclength(curve, Npoints) for curve in curves]

    cps_list        = make_symmetric_jelly_new(cps_list_og)     
    cps_list        = [hcat(cps, cps[:, 1]) for cps in cps_list]                 

    new_cps_list    = reverse_cps_list(cps_list)                 
    # new_cps_list    = [hcat(start .- shift[:,i], cps[:,2:end-1], ending .- shift[:,i]) for (i,cps) in enumerate(new_cps_list)]
    new_cps_list    = [hcat(start, cps[:,2:end-1], ending) for (i,cps) in enumerate(new_cps_list)]
    new_cps_list    = [SMatrix{2, 105, Float64, 210}(cps) for cps in new_cps_list]

    new_cps_list    = densify_cps_list(new_cps_list, 3.0; frames_per_segment=4)
return new_cps_list, opt_cps_list, cps_list_og, cps_list
end

function make_circle_cps_sin_motion(T::Type, npoints::Int, nsteps::Int; θ₁, radius=1.0, freq=0.1, amplitude=(0.25, 0.25))
    cps_list = Vector{SMatrix{2, npoints, T}}(undef, nsteps)

    θ = range(0, θ₁, length=npoints)

    for k in 0:nsteps-1
        shift = (amplitude[1] * sin(2π * freq * k), amplitude[2] * cos(2π * freq * k))
        x = radius .* cos.(θ) .+ shift[1]
        y = radius .* sin.(θ) .+ shift[2]
        cps_list[k+1] = SMatrix{2, npoints, T}(vcat(x', y')...)
    end

    return cps_list
end

function clamped_uniform_knots(p::Int, Ncp::Int)
    Ninterior = Ncp - p - 1
    head = zeros(Float64, p + 1)
    interior = Ninterior > 0 ? collect(range(0.0, 1.0, length = Ninterior + 2))[2:end-1] : Float64[]
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
    Plots.contour!(sim.flow.σ[R]'|>Array;levels,lines, title=title) 
    Plots.plot!(sim.body, add_cp=:true)    
end

@inline function interpolate_cps_hermite_new(new_cps_list, t::Tp, period; nphases::Int=10, smooth_tangent::Bool = false, tangent_scale=0.5) where Tp
    τ_total = t / period
    @show τ_total
    k = floor(Int, τ_total * nphases)
    τ_local = τ_total * nphases - k

    idx0 = mod(k, nphases) + 1
    idx1 = mod(k + 1, nphases) + 1
    idx_prev = mod(k - 1, nphases) + 1
    idx_next = mod(k + 2, nphases) + 1

    p_prev = new_cps_list[idx_prev]
    p0     = new_cps_list[idx0]
    p1     = new_cps_list[idx1]
    p_next = new_cps_list[idx_next]

    m0 = tangent_scale .* (p1 - p_prev)
    m1 = tangent_scale .* (p_next - p0)
    m0 = 0.5 .* (m0 .+ (p1 - p0))
    m1 = 0.5 .* (m1 .+ (p1 - p0))

    τ2 = τ_local^2
    τ3 = τ_local^3

    h00 = 2τ3 - 3τ2 + 1
    h10 = τ3 - 2τ2 + τ_local
    h01 = -2τ3 + 3τ2
    h11 = τ3 - τ2

    # return h00 .* p0 .+ h10 .* m0 .+ h01 .* p1 .+ h11 .* m1 , idx0
    interpolated = (1-τ_local) .* p0 .+ τ_local .* p1 #.+ SA{Tp}[D,2D] # Linear fallback 
    # interpolated = p0
    return interpolated, idx0
end

@inline function exp_interp(control_points_list, period, t)
    τ_scale = 1    # smaller = sharper, larger = smoother
    steps = length(control_points_list)
    Δt_frame = period / (steps - 1)
    τ = τ_scale * Δt_frame
    times = range(0, period; length=steps)
    k = searchsortedlast(times, t)
    k = clamp(k, 1, length(times)-1)
    t₀, t₁ = times[k], times[k+1]
    f₀, f₁ = control_points_list[k], control_points_list[k+1]
    γ = (1-expm1(-(t-t₀)/τ)) / (1-expm1(-(t₁ - t₀)/τ))
    return f₀ .+ (f₁ .- f₀) .* γ 
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

function simulate_Jelly!(sim, new_cps_list;
                         duration=1, period=3, step=0.1, verbose=true,
                         R=inside(sim.flow.p), remeasure=false, plotbody=false, kv...)

    Tp = eltype(sim.flow.p)
    t₀ = round(sim_time(sim))
    t = sum(sim.flow.Δt[1:end-1])
    period = period * sim.L / sim.U
    nphases = length(new_cps_list)

    forces_total = []
    forces_each_step = []
    div = []
    indices = []
    times = []
    # pressures_max = []
    # pressures_min = []

    idx_prev = 0
    n_cps = length(control_points_list[1][1,:])
    cps_paths_x = [[] for _ in 1:n_cps]  # vector of vectors
    cps_paths_y = [[] for _ in 1:n_cps]
    time_points = Float64[]
    t_geom = 0

    anim = @animate for tᵢ in range(t₀, t₀+duration; step)
        while t < tᵢ * sim.L / sim.U
            # @show t
            # sim.flow.Δt[end] = WaterLily.CFL(sim.flow; Δt_max=Tp(0.1))
            sim.flow.Δt[end] = Tp(0.1)
            cps_interp = exp_interp(new_cps_list, period, t_geom)
            # cps_interp, idx0 = interpolate_cps_hermite_new(new_cps_list, t_geom, period; nphases=Int(nphases))
            for (i, p) in enumerate(cps_interp[1,:])
                push!(cps_paths_x[i], p)
            end

            for (i, p) in enumerate(cps_interp[2,:])
                push!(cps_paths_y[i], p)
            end
            t_geom += 0.15
            @show t_geom
            # push!(time_points, Float64(t))

            body_interpolation = cps_interp .* 1 .* sim.L .+ (Tp(2sim.L), Tp(2.5sim.L))

            sim.sim.body = ParametricBodies.update!(sim.sim.body, body_interpolation, sim.flow.Δt[end])
            
            # if idx>idx_prev
            #     raw_total    = -WaterLily.total_force(sim)[1]
            #     push!(forces_total, raw_total)
            #     push!(indices,idx)
            #     idx_prev = idx
            # end
            
            raw_each_step    = -WaterLily.total_force(sim)[1]
            push!(forces_each_step, raw_each_step)
            push!(times, t)
            # push!(pressures_max, maximum(sim.flow.p))
            # push!(pressures_min, minimum(sim.flow.p))

            sim_step!(sim, tᵢ; remeasure)
            
            t += sim.flow.Δt[end]
        end
        
        save_dir = joinpath(pwd(), "Normals_check")
        isdir(save_dir) || mkpath(save_dir)

        x = 1:1:length(sim.flow.p[1,:])
        y = 1:1:length(sim.flow.p[1,:])
        nx, ny = length(x), length(y)

        xs, ys, nxs, nys = Float64[], Float64[], Float64[], Float64[]
        for j in 1:ny, i in 1:nx
            nvec = WaterLily.nds(sim.body, SVector(x[i], y[j]), 0.0)
            if norm(nvec) > 1e-6      
                push!(xs, x[i])
                push!(ys, y[j])
                push!(nxs, nvec[1])
                push!(nys, nvec[2])
            end
        end

        fig = Figure(resolution=(700,700))
        ax = Axis(fig[1,1], title="Surface Normals (nds)", aspect=DataAspect())

        arrows!(ax, xs, ys, nxs, nys, arrowsize=10, lengthscale=3, color=:blue)
        save(joinpath(save_dir, "nds_frame_$(tᵢ).png"))

        save_dir_p = joinpath(pwd(), "Pressure_check")
        isdir(save_dir_p) || mkpath(save_dir_p)

        WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))

        p = Array(sim.flow.p)        
        σ = Array(sim.flow.σ)          
        
        p_masked = copy(p)              
        p_masked[σ .< -ϵ] .= NaN         
        max_p = maximum(filter(!isnan, p_masked))
        min_p = minimum(filter(!isnan, p_masked))
        @show min_p, max_p
        pressure_plot = Plots.heatmap(p_masked', aspect_ratio=1,
        xlims=(1.5sim.L, 4sim.L),
        ylims=(1.5sim.L, 4sim.L),
        c=:balance,          
        clims=(-1, 1),  
        title="Velocity Field")

        Plots.contour!(sim.flow.σ',levels=[-sim.ϵ,0,sim.ϵ])   
        savefig(pressure_plot, joinpath(save_dir_p, "pressure_$(tᵢ).png"))

        @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L/sim.U
        @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I]) < 0.001, 0.0, sim.flow.σ[I])
        # @inside sim.flow.σ[I] = WaterLily.div(I,sim.flow.u)
        # @show maximum(abs,sim.flow.σ[R]|>Array)
        # push!(div, maximum(abs,sim.flow.σ[R]|>Array))
        flood(sim.flow.σ[R] |> Array; clims=(-0.5,0.5), kv...)

        plotbody && body_plot!(sim; title="$(round(t, digits=4))")

        verbose && println("t=", round(t, digits=4), ", Δt=", round(sim.flow.Δt[end], digits=5))
    end 

    gif(anim,"Swimming_Jelly.gif")

    return (forces=forces_each_step, cps_paths_x=cps_paths_x, cps_paths_y=cps_paths_y, time=times)
end

include("New_Simu_Trial.jl")

# new_cps_list, opt_cps_list, cps_list_og, cps_list = create_cps_list(Float64)
set_times = [1,10,20,30,40,50,60,70,80,90]

plt = Plots.scatter(new_cps_list[5][1,:], new_cps_list[5][2,:])
for i in set_times
    plt_2 = Plots.scatter!(new_cps_list[i][1,:], new_cps_list[i][2,:])
    display(plt_2)
end
display(plt)
# new_cps_list = get_cps_list(cps_optimizer()[1], 3, 10)
# new_cps_list    = make_circle_cps_sin_motion(Float64, 22, 10; θ₁=2π, radius=0.25,freq=0.1, amplitude=(0.25, 0.25))

# num_timesteps = length(new_cps_list)
# num_cps = size(new_cps_list[1], 2)  # Number of control points
# velocities = zeros(num_cps, num_timesteps - 1)

# # Compute velocity magnitudes per control point per timestep
# for t in 1:num_timesteps-1
#     for j in 1:num_cps
#         pos_now = new_cps_list[t][:, j]
#         pos_next = new_cps_list[t+1][:, j]
#         velocities[j, t] = norm(pos_next - pos_now)
#     end
# end
# plt = Plots.scatter(new_cps_list[1][1,:], new_cps_list[1][2,:], xlims=(0,1.75), legend=:right)
# Plots.scatter!(new_cps_list[111][1,:], new_cps_list[111][2,:])
# Plots.scatter!(new_cps_list[91][1,:], new_cps_list[91][2,:])
# display(plt)

# # Plot: each control point gets its own velocity curve over time
# plt = Plots.plot(title="Velocity per Control Point", xlabel="Time Step", ylabel="Velocity Magnitude")
# for j in 1:num_cps
#     Plots.plot!(1:num_timesteps-1, velocities[j, :], label="CP $j")
# end
# display(plt)

# const v = 0.0f0

D = 2^5; Re = 302; U = 1; ϵ = 0.75; thk = 2ϵ+√3; deg = 2; cycles = Tp(1); period = Tp(3); duration = cycles * period                                                     
sim             = TwoDimJellyfish(; new_cps_list, D, Re, U, ϵ, thk, deg, mem=Array, use_biotsavart=true) 

# WaterLily.logger("test_psolver")
res             = simulate_Jelly!(sim, new_cps_list; duration, period, step = 0.1, remeasure = true, plotbody = true)
# plot_logger("test_psolver")
# savefig("psolver.png")





struct CFD_Jellyfish{T=Float32}
    # Physical Parameters
    Re::T           # Reynolds Number
    U::T            # Far-field velocity
    period::T       # Period of motion cycle
    cycles::T       # Number of cycles to be calculated

    # Numerical Parameters
    D::T            # Grid Size
    ϵ::T            # Boundary Layer Thickness
    deg::T          # Polynomial Degree of Jellyfish NURBS curve
end
CFD_Jellyfish(Re, U, period, cycles, D, ϵ, deg) = 
    CFD_Jellyfish{Float32}(Re, U, period, cycles, D, ϵ, deg)

@inline function TwoDimJellyfish_new(case::CFD_Jellyfish, cps_list; mem=Array, use_biotsavart=false)
    T = typeof(case.U)
    thk = 2case.ϵ + √3
    ν = case.U * case.D / case.Re
    cps = cps_list[1] .* case.D .+ SA{T}[2case.D, 2.5case.D]

    weights = ones(T, size(cps, 2))
    knots = T.(clamped_uniform_knots(case.deg, size(cps, 2)))
    curve = NurbsCurve(cps, knots, weights)

    jelly_body = DynamicNurbsBody(curve; thk=thk, boundary=true)

    return use_biotsavart ?
        BiotSimulation((6case.D, 6case.D), (0,0), case.D; case.U, ν, jelly_body, T, mem, case.ϵ) :
        Simulation((6case.D, 6case.D), (0,0), case.D; case.U, ν, jelly_body, T, mem, case.ϵ)
end

case = CFD_Jellyfish(302.0, 1.0, 1.0, 3.0, 2^7, 0.5, 3)
sim = build_simulation(case; cps_list=my_cps)

struct JellyRunConfig{T<:AbstractFloat}
    duration::T        # total duration of simulation
    period::T          # motion period
    step::T            # animation timestep
    verbose::Bool      # print progress
    remeasure::Bool    # recompute geometry each step
    plotbody::Bool     # plot geometry each frame
    save_normals::Bool # whether to save surface normals
    save_pressure::Bool # whether to save pressure field
end
JellyRunConfig(; duration=1.0, period=3.0, step=0.1, 
                verbose=true, remeasure=false, plotbody=false,
                save_normals=true, save_pressure=true) = 
                JellyRunConfig(duration, period, step, verbose, remeasure, plotbody, save_normals, save_pressure)

function simulate!(sim, new_cps_list; config=JellyRunConfig(), kv...)
    cfg = config
    Tp = eltype(sim.flow.p)
    t₀ = round(sim_time(sim))
    t = sum(sim.flow.Δt[1:end-1])
    nphases = length(new_cps_list)
    period_scaled = cfg.period * sim.L / sim.U

    forces_total = Float64[]

    anim = @animate for tᵢ in range(t₀, t₀ + cfg.duration; step=cfg.step)
        while t < tᵢ * sim.L / sim.U
            sim.flow.Δt[end] = WaterLily.CFL(sim.flow; Δt_max=Tp(0.05))

            cps_interp = interpolate_cps_hermite_new(new_cps_list, t, period_scaled; nphases)
            body_interp = cps_interp .* sim.L .+ SA[2sim.L, 2.5sim.L]
            sim.sim.body = ParametricBodies.update!(sim.sim.body, body_interp, sim.flow.Δt[end])

            push!(forces_total, -WaterLily.total_force(sim)[1])
            sim_step!(sim, tᵢ; cfg.remeasure)

            t += sim.flow.Δt[end]
        end

        cfg.save_normals && save_normals_plot(sim, tᵢ)
        cfg.save_pressure && save_pressure_plot(sim, tᵢ)

        cfg.plotbody && body_plot!(sim; title="$(round(t, digits=4))")
        cfg.verbose && @info "t=$(round(t, digits=4)), Δt=$(round(sim.flow.Δt[end], digits=5))"
    end

    gif(anim, "Swimming_Jelly.gif")
    return forces_total
end


cfg = JellyRunConfig(duration=1.0, period=3.0, verbose=false, plotbody=true)
forces = simulate!(sim, my_cps; config=cfg)



function save_normals_plot(sim, tᵢ)
    save_dir = mkpath(joinpath(pwd(), "Normals_check"))
    x = 1:length(sim.flow.p[1, :])
    y = 1:length(sim.flow.p[:, 1])
    xs, ys, nxs, nys = Float64[], Float64[], Float64[], Float64[]
    for j in y, i in x
        nvec = WaterLily.nds(sim.body, SVector(i, j), 0.0)
        if norm(nvec) > 1e-6
            push!(xs, i); push!(ys, j)
            push!(nxs, nvec[1]); push!(nys, nvec[2])
        end
    end
    fig = Figure(resolution=(700, 700))
    ax = Axis(fig[1, 1], title="Surface Normals", aspect=DataAspect())
    arrows!(ax, xs, ys, nxs, nys, arrowsize=10, lengthscale=3, color=:blue)
    save(joinpath(save_dir, "nds_frame_$(tᵢ).png"))
end

function save_pressure_plot(sim, tᵢ; ϵ=sim.ϵ)
    save_dir = mkpath(joinpath(pwd(), "Pressure_check"))
    WaterLily.measure_sdf!(sim.flow.σ, sim.body, WaterLily.time(sim))
    p, σ = Array(sim.flow.p), Array(sim.flow.σ)
    p_masked = copy(p)
    p_masked[σ .< -ϵ] .= NaN
    heat = Plots.heatmap(p_masked', aspect_ratio=1, c=:balance,
        xlims=(1.75sim.L, 4sim.L), ylims=(1.75sim.L, 3.5sim.L),
        clims=(-1, 1), title="Pressure Field")
    Plots.contour!(sim.flow.σ', levels=[-sim.ϵ, 0, sim.ϵ])
    savefig(heat, joinpath(save_dir, "pressure_$(tᵢ).png"))
end