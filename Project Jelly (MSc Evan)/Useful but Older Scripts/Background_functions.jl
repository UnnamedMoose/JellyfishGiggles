function exp_smooth(x::Vector{Float64}, α::T) where {T<:Real}
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

function smooth_spline_variable_upsample(cps_seq; s=0.001, up1=10, up2=5, split=nothing)
    Nframes = length(cps_seq)
    @show Nframes
    Npts    = size(cps_seq[1], 2)
    t       = 1:Nframes
    mid     = split === nothing ? (Nframes ÷ 2) : split

    t1 = range(first(t), mid; length = (mid - first(t)) * up1 + 1)
    t2 = range(mid, last(t);  length = (last(t) - mid) * up2 + 1)
    @show  (mid - first(t)) * up1 + 1
    @show  (last(t) - mid) * up2 + 1
    t_interp = [t1; t2[2:end]]  # drop duplicate 'mid'

    splx = Vector{Spline1D}(undef, Npts)
    sply = Vector{Spline1D}(undef, Npts)
    for i in 1:Npts
        xs = [cps_seq[k][1,i] for k in t]
        ys = [cps_seq[k][2,i] for k in t]
        splx[i] = Spline1D(t, xs; k=3, s=s)
        sply[i] = Spline1D(t, ys; k=3, s=s)
    end

    out = Vector{SMatrix{2, Npts, Float64}}(undef, length(t_interp))
    for (j, τ) in enumerate(t_interp)
        M = Matrix{Float64}(undef, 2, Npts)
        for i in 1:Npts
            M[1,i] = splx[i](τ)
            M[2,i] = sply[i](τ)
        end
        out[j] = SMatrix{2, Npts, Float64, 100}(M)
    end
    return out
end

function control_point_functions(sx, sy, t_points)
    N = length(sx)
    interp_funcs = Vector{Function}(undef, N)
    for i in 1:N

        fx = Spline1D(t_points, sx[i], k=4)
        fy = Spline1D(t_points, sy[i], k=4)

        interp_funcs[i] = t -> SA[fx(t), fy(t)]
    end
    return interp_funcs
end

function cps_at_time(interp_funcs, Npoints, t; ThreeD=false)
    if ThreeD == true
        # Npoints = 53
        zcoords = zeros(1, Npoints)
        cps_xy = SMatrix{2,Npoints,Float64}(hcat([f(t) for f in interp_funcs]...) )
        cps_t  = SMatrix{3,Npoints,Float64}(vcat(cps_xy, zcoords))
    else
        cps_t = SMatrix{2,Npoints,Float64}(hcat([f(t) for f in interp_funcs]...) )
    end
    return cps_t
end

function blend_cycles(v::Vector{Float64}, n_cycles::Int; overlap_ratio=0.1)
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

"""
The following function constructs the jellyfish motion, returning a list of control point matrices.
Ncps = number of control points to define half of the jellyfish.
spline_s = smoothness for defining the control points list based on the control point trajectories that are defined using a 1D Dierckx spline.
path_s = second smoothing parameter to define the smoothing for the exponential smoothing algorithm.
up = the number of samples between each initial frame.
deg = polynomial degree.
γ = expansion phase duration / contraction phase duration.
λ_area = area conservation coefficient.
λ_shape = shape conservation coefficient.

Uses the following functions during computation:
resample_by_arclength
BSplineCurve
smooth_spline_variable_upsample
optimize_control_points
get_area
"""

function construct_jelly_motion(Ncps::Int=50,spline_s::Float64=0.001, up::Int=15, deg::Any=2; γ::Int=2, λ_area::T=T(1e-1), λ_shape::T=T(1e-3))
    cps_0   = SA{T}[0.000  0.024  0.064  0.175  0.323  0.481  0.643  0.798  0.996  1.191  1.228  1.201  1.171  1.178  1.154  0.804  0.606  0.475  0.397  0.340  0.323  
                    0.000  0.193  0.333  0.475  0.578  0.623  0.627  0.614  0.571  0.524  0.484  0.213  0.216  0.412  0.456  0.451  0.420  0.348  0.266  0.151 0.000  ] 
    cps_1   = SA{T}[0.000  0.024  0.067  0.175  0.326  0.488  0.639  0.794  0.993  1.164  1.198  1.296  1.265  1.171  1.090  0.801  0.606  0.478  0.404  0.343  0.323  
                    0.000  0.193  0.326  0.469  0.564  0.609  0.606  0.590  0.537  0.501  0.470  0.216  0.206  0.412  0.426  0.438  0.396  0.335  0.256  0.154  0.000  ] 
    cps_2   = SA{T}[0.000  0.024  0.081  0.188  0.333  0.481  0.639  0.798  0.986  1.150  1.222  1.390  1.373  1.205  1.154  0.801  0.616  0.501  0.427  0.370  0.323  
                    0.000  0.193  0.319  0.455  0.547  0.589  0.583  0.550  0.473  0.400  0.350  0.173  0.159  0.304  0.334  0.380  0.366  0.311  0.243  0.148 0.000  ] 
    cps_3   = SA{T}[0.000  0.034  0.091  0.199  0.337  0.481  0.629  0.781  0.973  1.077  1.181  1.346  1.319  1.198  1.154  0.798  0.643  0.522  0.448  0.387  0.357  
                    0.000  0.193  0.322  0.448  0.541  0.568  0.566  0.533  0.449  0.412  0.392  0.149  0.135  0.291  0.301  0.343  0.342  0.287  0.226  0.141  0.000  ]    
    cps_4   = SA{T}[0.000  0.027  0.081  0.199  0.357  0.478  0.626  0.781  0.976  1.178  1.222  1.228  1.191  1.195  1.151  0.801  0.633  0.525  0.438  0.384  0.347  
                    0.000  0.204  0.326  0.466  0.562  0.589  0.587  0.560  0.507  0.467  0.437  0.200  0.206  0.359  0.389  0.381  0.363  0.315  0.233  0.158  0.000  ] 
    cps_5   = SA{T}[0.000  0.027  0.074  0.182  0.340  0.471  0.629  0.791  0.989  1.185  1.222  1.151  1.124  1.164  1.127  0.798  0.616  0.498  0.407  0.360  0.337  
                    0.000  0.204  0.340  0.483  0.585  0.620  0.624  0.608  0.561  0.518  0.478  0.237  0.250  0.420  0.454  0.452  0.410  0.352  0.253  0.172  0.000  ] 
    cps_6   = SA{T}[0.000  0.027  0.074  0.182  0.330  0.464  0.629  0.794  0.989  1.185  1.225  1.158  1.124  1.164  1.141  0.794  0.613  0.488  0.394  0.343  0.330  
                    0.000  0.204  0.340  0.483  0.592  0.637  0.641  0.628  0.592  0.532  0.495  0.233  0.250  0.403  0.454  0.469  0.424  0.366  0.267  0.175  0.000  ] 
    cps_7   = SA{T}[0.000  0.027  0.074  0.182  0.330  0.464  0.629  0.794  0.989  1.185  1.225  1.178  1.144  1.164  1.141  0.794  0.613  0.488  0.394  0.343  0.330  
                    0.000  0.204  0.340  0.483  0.592  0.637  0.641  0.628  0.592  0.532  0.495  0.227  0.230  0.403  0.454  0.469  0.424  0.366  0.267  0.175  0.000  ] 
    cps_8   = SA{T}[0.000  0.027  0.074  0.182  0.330  0.464  0.629  0.794  0.989  1.185  1.225  1.185  1.151  1.164  1.141  0.794  0.613  0.488  0.394  0.343  0.330  
                    0.000  0.204  0.340  0.483  0.592  0.637  0.641  0.628  0.592  0.532  0.495  0.230  0.233  0.403  0.454  0.469  0.424  0.366  0.267  0.175  0.000  ] 
    cps_9   = SA{T}[0.000  0.027  0.074  0.182  0.330  0.464  0.629  0.794  0.989  1.185  1.225  1.191  1.154  1.164  1.141  0.794  0.613  0.488  0.394  0.343  0.330  
                    0.000  0.204  0.340  0.483  0.592  0.637  0.641  0.628  0.592  0.532  0.495  0.220  0.220  0.403  0.454  0.469  0.424  0.366  0.267  0.175  0.000  ] 
    start   = SA{T}[0.000 0.000 0.000 0.000; 0.000 -0.010 -0.020 -0.030]
    ending  = SA{T}[0.000 0.000 0.000 0.000; 0.030  0.020  0.010  0.000]

    cps_contr           = [cps_0, cps_1, cps_2, cps_3, cps_4, cps_5]
    cps_exp             = [cps_5, cps_6, cps_7, cps_8, cps_9, cps_0]

    cps_list_contr      = Vector{SMatrix{2,50,Float64,100}}([resample_by_arclength(BSplineCurve(to_static(cps[:,1:end]); degree=deg), Ncps) for cps in cps_contr])
    cps_list_exp        = Vector{SMatrix{2,50,Float64,100}}([resample_by_arclength(BSplineCurve(to_static(cps[:,1:end]); degree=deg), Ncps) for cps in cps_exp])   
    cps_contr           = Vector{SMatrix{2,50,Float64,100}}(vcat(smooth_spline_variable_upsample(cps_list_contr; s=spline_s, up1=up, up2=up, split=nothing) ))
    cps_exp             = Vector{SMatrix{2,50,Float64,100}}(vcat(smooth_spline_variable_upsample(cps_list_exp; s=spline_s, up1=Int(up*γ), up2=Int(up*γ), split=nothing) ))
    
    cps_contr           = cps_contr |> make_symmetric_jelly_new |> x -> [hcat(cp, cp[:, 1]) for cp in x] |> reverse_cps_list |>
                            x -> [hcat(start, cps[:, 2:end-1], ending) for (i, cps) in enumerate(x)] |>
                            x -> [SMatrix{2,105,Float64,210}(cps) for cps in x]

    cps_exp             = cps_exp |> make_symmetric_jelly_new |> x -> [hcat(cp, cp[:, 1]) for cp in x] |> reverse_cps_list |>
                            x -> [hcat(start, cps[:, 2:end-1], ending) for (i, cps) in enumerate(x)] |>
                            x -> [SMatrix{2,105,Float64,210}(cps) for cps in x]

    cps_list_new        = Vector{SMatrix{2,105,Float64,210}}(vcat(cps_contr, cps_exp))
    # cps_list_new        = Vector{SMatrix{2,105,Float64,210}}([optimize_control_points(cps, get_area(cps_list_new[1]); λ_area=λ_area, λ_shape=λ_shape, degree=deg) for cps in cps_list_new])

    n_cps = length(cps_list_new[1][1,:]); cps_paths_x = [[] for _ in 1:n_cps]; cps_paths_y = [[] for _ in 1:n_cps]
    
    for j in 1:length(cps_list_new)
        for (i, p) in enumerate(cps_list_new[j][1,:])
            push!(cps_paths_x[i], p)
        end
        for (i, p) in enumerate(cps_list_new[j][2,:])
            push!(cps_paths_y[i], p)
        end
    end

    cps_paths_x         = Vector{Vector{T}}(cps_paths_x) 
    cps_paths_y         = Vector{Vector{T}}(cps_paths_y)
    return cps_paths_x, cps_paths_y
end

function gen_p_plots(sim, tᵢ)
    save_dir_p = joinpath("Prolate Jellyfish", "Pressure_check")
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

function gen_n_plots(sim, tᵢ)
    save_dir = joinpath("Prolate Jellyfish", "Normals_check")
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
end

function gen_ω_gif(sim, t, R)
    save_dir_ω = joinpath("Prolate Jellyfish", "Vorticity_check")
    isdir(save_dir_ω) || mkpath(save_dir_ω)
    @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L/sim.U
    @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I]) < 0.001, 0.0, sim.flow.σ[I])
    # vorticity_plot = flood(sim.flow.σ[R] |> Array; clims=(-1, 1))
    vorticity_plot = flood(sim.flow.σ,shift=(-1.5,-1.5),clims=(-1,1),axis=([],false),
              cfill=:seismic,legend=false,border=:none)
    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
    Plots.contour!(sim.flow.σ',levels=[-sim.ϵ,0,sim.ϵ], title="$(round(t, digits=4))")
    savefig(vorticity_plot, joinpath(save_dir_ω, "vorticity_$(t).png"))
end

function gen_div_gif(sim, t, R)
    @inside sim.flow.σ[I] = WaterLily.div(I,sim.flow.u)
    @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I]) < 0.001, 0.0, sim.flow.σ[I])
    WaterLily.flood(sim.flow.σ[R] |> Array; clims=(-0.5, 0.5))
    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
    Plots.contour!(sim.flow.σ',levels=[-sim.ϵ,0,sim.ϵ], title="$(round(t, digits=4))")
end

function create_gif_from_folder(folder_path::String, output_path::String; delay::Float64=0.1)
    image_files = sort(filter(f -> any(ext -> endswith(lowercase(f), ext), [".png", ".jpg", ".jpeg"]), readdir(folder_path, join=true)))

    function extract_float(path)
        m = match(r"([0-9]+(?:\.[0-9]+)?) (?= \.\w+$)"x, path)
        return m === nothing ? Inf : parse(Float64, m.captures[1])
    end
    sorted_files = sort(image_files, by=extract_float)

    frames = [load(f) for f in sorted_files]

    save(output_path, cat(frames...; dims=3); fps=1/delay)
    println("GIF saved to: $output_path")
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

    fixed_pt = cps[:, 1:3]  # first and last point (fixed)
    last_pt = cps[:, 11:end]
    fixed_pts = cps[:, ]
    # n_inner = N - 2       # number of inner points

    # Vectorize inner control points only
    # x0_inner = vec(Matrix(cps[:, 2:N-1]))  # 2*(N-2) vector
    x0_inner =  vec(Matrix(cps[:, 4:10]))
    n_inner = length(x0_inner) ÷ 2
    s_vals = range(0, stop=1, length=nsamples)

    cost = function (x::AbstractVector)
        X_inner = reshape(x, 2, n_inner)
        # X_inner[2, end] = 0.0  # This sets y-value to 0
        
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

to_static(cps::AbstractMatrix{T}) where {T} = SMatrix{size(cps,1), size(cps,2), T}(cps)

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

function visualize_sdf_3D(body; D=2^7, n=75, T=Float32,
                          surface_only=true,
                          colormap=:viridis,
                          zero_color=:red)

    xs = range(0D, 3D; length=n)
    ys = range(0D, D; length=n)
    zs = range(0D, D; length=n)

    φ = [sdf(body, SA[T(x), T(y), T(z)]) for x in xs, y in ys, z in zs]

    fig = Figure(resolution=(900,700))
    ax = Axis3(fig[1,1],
        title = "Signed Distance Field",
        perspectiveness = 0.9
    )

    xside = xs[1] .. xs[end]
    yside = ys[1] .. ys[end]
    zside = zs[1] .. zs[end]

    if surface_only
        GLMakie.contour!(
            ax, xside, yside, zside, φ;
            levels=[0.0],
            colormap = :plasma,
            linewidth = 1.0
        )

    else
        # Volume plot (important: set colorrange manually!)
        GLMakie.volume!(
            ax, xside, yside, zside, φ;
            colormap = colormap,
            colorrange = extrema(φ),
            transparency = true,
            alpha = 0.5
        )

        # Zero-level surface
        GLMakie.contour!(
            ax, xside, yside, zside, φ;
            levels=[0.0],
            colormap = [zero_color],
            alpha=1.0
        )

        Colorbar(fig[1,2], colormap=colormap, limits=extrema(φ))
    end

    GLMakie.xlims!(ax, xs[1], xs[end])
    GLMakie.ylims!(ax, ys[1], ys[end])
    GLMakie.zlims!(ax, zs[1], zs[end])

    ax.aspect = :data
    fig
end

function visualise_3D_Jelly(sim, θ₂)
    X = [sim.body.map(sim.body.curve(u), θ)[1] for u in LinRange(0, 1, 80), θ in LinRange(0, θ₂, 80)]
    Y = [sim.body.map(sim.body.curve(u), θ)[2] for u in LinRange(0, 1, 80), θ in LinRange(0, θ₂, 80)]
    Z = [sim.body.map(sim.body.curve(u), θ)[3] for u in LinRange(0, 1, 80), θ in LinRange(0, θ₂, 80)]

    fig = Figure(resolution = (900, 700))
    ax = Axis3(fig[1, 1], title = "Revolved NURBS Jellyfish")
    GLMakie.surface!(ax, X, Y, Z, colormap = :Blues, shading = true)
    ax.xlabel = "x"
    ax.ylabel = "y"
    ax.zlabel = "z"
    fig
end

function mirrorto!(a,b)
    nx, ny, nz = size(b)

    # Fill quadrants from original block b (never from a)
    @views a[:, ny+1:2ny, nz+1:2nz]   .= b                    # y+ , z+
    @views a[:, 1:ny, nz+1:2nz] .= b[:, ny:-1:1, :]     # y− , z+
    @views a[:, ny+1:2ny,   1:nz] .= b[:, :, nz:-1:1]   # y+ , z−
    @views a[:, 1:ny, 1:nz] .= b[:, ny:-1:1, nz:-1:1] # y− , z−

    return a
end

function geom!(md,d,sim,t=WaterLily.time(sim))
    a = sim.flow.σ
    WaterLily.measure_sdf!(a,sim.body,t)
    copyto!(d,a[inside(a)]) # copy to CPU
    mirrorto!(md,d)         # mirror quadrant
    @show size(md)
    alg = Meshing.MarchingCubes()
    ranges = range.((0, 0, 0), size(md))
    points, faces = Meshing.isosurface(md, alg, ranges...)
    p3f = Point3f.(points)
    gltriangles = GLMakie.GLTriangleFace.(faces)
    return GLMakie.normal_mesh(p3f, gltriangles)
end

function ω!(md,d,sim)
    a,dt = sim.flow.σ,sim.L/sim.U
    @inside a[I] = WaterLily.ω_mag(I,sim.flow.u)*dt
    copyto!(d,a[inside(a)]) # copy to CPU
    @show d
    mirrorto!(md,d)         # mirror quadrant
end