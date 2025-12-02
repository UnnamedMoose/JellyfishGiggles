using LinearAlgebra, Printf, Statistics, Plots, ParametricBodies, StaticArrays
using WaterLily, CUDA
using GeometryBasics, Optim
using BiotSavartBCs
using Interpolations: LinearInterpolation
using DelimitedFiles, DataFrames, CSV
using Dierckx

import WaterLily: CFL
CFL(a::Flow;Δt_max=10) = 0.1

import WaterLily: sim_step!
function sim_step!(sim::AbstractSimulation,t_end;remeasure=true,λ=quick,max_steps=typemax(Int),verbose=false,
        udf=nothing,kwargs...)
    steps₀ = length(sim.flow.Δt)
    while sim_time(sim) < t_end && length(sim.flow.Δt) - steps₀ < max_steps
        sim_step!(sim; remeasure, λ, udf, kwargs...)
        verbose && sim_info(sim)
    end
end

Tp = Float64; T = Float64

@info "Running with $(Threads.nthreads()) Julia threads"

function exp_smooth(x::Vector{Float64}, α::T) where {T<:Real}
    s₀ = similar(x)     
    s₀[1] = x[1]
    for t in 2:length(x)
        s₀[t] = α * x[t] + (1 - α) * s₀[t-1]
    end
    s₁ = similar(x)    
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
    t_interp = [t1; t2[2:end]]  

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

function cps_at_time(interp_funcs, Npoints, t)
    cps_t = SMatrix{2,Npoints,Float64}(hcat([f(t) for f in interp_funcs]...) )
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

function construct_jelly_motion(Ncps=50,spline_s=0.001, up=15; ThreeD=false)
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
    start   = SA{T}[0.000 0.000 0.000 0.000;
                0.000 -0.010 -0.020 -0.030]
    ending  = SA{T}[0.000 0.000 0.000 0.000;
                0.030 0.020 0.010 0.000]
    cps_contr       = [cps_0, cps_1, cps_2, cps_3, cps_4, cps_5]
    cps_exp        = [cps_5, cps_6, cps_7, cps_8, cps_9, cps_0]
    cps_list_contr  = Vector{SMatrix{2,50,Float64,100}}([resample_by_arclength(BSplineCurve(to_static(cps[:,1:end]); degree=2), Ncps) for cps in cps_contr])
    cps_list_exp   = Vector{SMatrix{2,50,Float64,100}}([resample_by_arclength(BSplineCurve(to_static(cps[:,1:end]); degree=2), Ncps) for cps in cps_exp])   
    cps_contr       = Vector{SMatrix{2,50,Float64,100}}(vcat(smooth_spline_variable_upsample(cps_list_contr; s=spline_s, up1=up*2, up2=up*2, split=nothing) ))
    cps_exp        = Vector{SMatrix{2,50,Float64,100}}(vcat(smooth_spline_variable_upsample(cps_list_exp; s=spline_s, up1=up*4, up2=up*4, split=nothing) ))
    cps_contr = cps_contr |> make_symmetric_jelly_new |> x -> [hcat(cp, cp[:, 1]) for cp in x] |> reverse_cps_list |>
    x -> [hcat(start, cps[:, 2:end-1], ending) for (i, cps) in enumerate(x)] |>
    x -> [SMatrix{2,105,Float64,210}(cps) for cps in x]
    cps_exp = cps_exp |> make_symmetric_jelly_new |> x -> [hcat(cp, cp[:, 1]) for cp in x] |> reverse_cps_list |>
    x -> [hcat(start, cps[:, 2:end-1], ending) for (i, cps) in enumerate(x)] |>
    x -> [SMatrix{2,105,Float64,210}(cps) for cps in x]
    cps_list_new = Vector{SMatrix{2,105,Float64,210}}(vcat(cps_contr, cps_exp))
    cps_list_new = Vector{SMatrix{2,105,Float64,210}}([optimize_control_points(cps, get_area(cps_list_new[1])) for cps in cps_list_new])

    ThreeD && (new_cps_list = [SMatrix{3, size(cps,2)}(vcat(cps, zeros(1, size(cps,2)))) for cps in cps_const])

    n_cps = length(cps_list_new[1][1,:]); cps_paths_x = [[] for _ in 1:n_cps]; cps_paths_y = [[] for _ in 1:n_cps]
    for j in 1:length(cps_list_new)
        for (i, p) in enumerate(cps_list_new[j][1,:])
            push!(cps_paths_x[i], p)
        end
        for (i, p) in enumerate(cps_list_new[j][2,:])
            push!(cps_paths_y[i], p)
        end
    end
    cps_paths_x = Vector{Vector{T}}(cps_paths_x); cps_paths_y = Vector{Vector{T}}(cps_paths_y)
    return cps_paths_x, cps_paths_y, cps_contr, cps_exp, cps_list_new
end

function gen_p_plots(sim, tᵢ)
    save_dir_p = joinpath("Prolate Jellyfish", "Pressure_check")
    isdir(save_dir_p) || mkpath(save_dir_p)

    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))

    p = Array(sim.flow.p)        
    σ = Array(sim.flow.σ)          
    
    p_masked = copy(p)              
    p_masked[σ .< -ϵ] .= NaN         
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

    xs, ys, nxs, nys = Float64[], Float64[], Float64[], Float64[]
    p_masked = fill(NaN, nx, ny)
    for j in 1:ny, i in 1:nx
        nvec = WaterLily.nds(sim.body, SVector(x[i], y[j]), 0.0)
        if norm(nvec) > 1e-6       
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
    vorticity_plot = flood(sim.flow.σ[R] |> Array; clims=(-1, 1))
    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
    Plots.contour!(sim.flow.σ',levels=[-sim.ϵ,0,sim.ϵ], title="$(round(t, digits=4))")
    savefig(vorticity_plot, joinpath(save_dir_ω, "vorticity_$(t).png"))
end

function gen_div_gif(sim, t, R)
    @inside sim.flow.σ[I] = WaterLily.div(I,sim.flow.u)
    @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I]) < 0.001, 0.0, sim.flow.σ[I])
    flood(sim.flow.σ[R] |> Array; clims=(-0.5, 0.5))
    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
    Plots.contour!(sim.flow.σ',levels=[-sim.ϵ,0,sim.ϵ], title="$(round(t, digits=4))")
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

ThreeD      = false             
D           = 2^5               
Re          = 302               
St          = 0.52              
U           = 1                 
ϵ           = 1                 
thk         = 2ϵ+√3            
deg         = 2                
cycles      = 5                 
period      = (D/U) / St       
duration    = cycles * period   
path_x, path_y, cps_contr, cps_exp, cps_list_new    = construct_jelly_motion(50,0.001,5; ThreeD=ThreeD)
path_x                                              = [blend_cycles(p, 5) for p in path_x]
path_y                                              = [blend_cycles(p, 5) for p in path_y]
len                                                 = length(path_x)
path_x_smooth                                       = [exp_smooth(path_x[i], 0.250) for i in 1:len]
path_y_smooth                                       = [exp_smooth(path_y[i], 0.250) for i in 1:len]  
frame_points                                        = range(1,length(path_x_smooth[25]), step=1)
pathing                                             = control_point_functions(path_x_smooth, path_y_smooth, frame_points)

cps_start = cps_at_time(pathing, 105, 0) 

@inline function TwoDimJellyfish(::Type{T}=Float32; new_cps_list, D=2^7, Re=302, U=1, ϵ=0.5, thk=2ϵ+√3, deg, mem=Array, use_biotsavart=false) where {T<:AbstractFloat}
    ν           =   U * D / Re

    cps         =   new_cps_list .* 1 .* D .+ SA{T}[3.5D, 2.5D]
    degree      =   deg
    n_ctrl      =   size(cps, 2)
    weights     =   ones(T, n_ctrl)
    knots       =   T.(clamped_uniform_knots(degree, n_ctrl))
    curve       =   NurbsCurve(cps, knots, weights)
    body        =   DynamicNurbsBody(curve; thk=thk, boundary=true)

    return use_biotsavart ? BiotSimulation((6D, 6D), (0,0), D; U, ν, body, T, mem, ϵ) : Simulation((6D, 6D), (0,0), D; U, ν, body, T, mem, ϵ)
end

function run_jelly_simulation(cps_start, D, Re, U, ϵ, thk, deg, pathing)
    sim         = TwoDimJellyfish(; new_cps_list=cps_start, D, Re, U, ϵ, thk, deg, mem=Array, use_biotsavart=true)
    forces      = []; forces_filt = []; forces_out = []; time = []; time_sim = []; timesteps = []; displacement = []; velocity = []; acceleration = []
    n_cps       = length(cps_start)
    cps_paths_x = [[] for _ in 1:n_cps]
    prev_force  = 0
    duration    = 2; t₀ = round(sim_time(sim)); step = 0.1
    t0 = 0; a0 = 0; v0 = 0; p0 = 0; Area = get_area(cps_start .* sim.L)
    hₛ = 0.85*D; dₛ=0.8*D; d=1.2*D; h=1.3*D
    mₐ = (2*h / d)^1.4 * (π * dₛ^2 * hₛ) / (6)

    for tᵢ in range(t₀, t₀ + duration; step)        
        t = sum(sim.flow.Δt[1:end-1])
        while t < tᵢ * sim.L / sim.U
            cps             = cps_at_time(pathing, 105, t) .* D .+ SA{T}[3.5D+p0, 2.5D]

            sim.sim.body    = ParametricBodies.update!(sim.sim.body, cps, sim.flow.Δt[end])
            sim_step!(sim, t/sim.L; remeasure = true)
            
            force           =   -WaterLily.pressure_force(sim)[1]
            filt_force      =   0.1 * force + (1-0.1) * prev_force
            Δt              =   sim.flow.Δt[end]
            accel           =   (filt_force + mₐ * a0) / (Area + mₐ)
            p0              +=  Δt * (v0 + Δt * accel / 2.)
            v0              +=  Δt * accel
            a0              =   accel

            push!(velocity, v0)
            push!(displacement, p0)
            push!(acceleration, a0)
            push!(timesteps, sim.flow.Δt[end])
            push!(time, t * sim.U / sim.L)
            push!(forces, force)
            push!(forces_filt, filt_force)
            push!(time_sim, sim_time(sim))
            for (i, p) in enumerate(cps[1, :])
                push!(cps_paths_x[i], p)
            end

            t0 = t; t += sim.flow.Δt[end]; prev_force = filt_force
        end

        force_out   =   -WaterLily.pressure_force(sim) / (sim.L * 0.5)
        R           =   inside(sim.flow.p)

        gen_p_plots(sim, tᵢ)
        gen_ω_gif(sim, tᵢ, R)
        push!(forces_out, force_out[1]) 
        println("tU/L=", round(tᵢ, digits = 4), ", Δt=", round(sim.flow.Δt[end], digits = 3))
    end 

    return forces, forces_out, forces_filt, time, time_sim, timesteps, cps_paths_x, displacement, velocity, acceleration
end

WaterLily.logger("test_psolver")
forces, force_out, force_filt, time, time_sim, timesteps, cps_paths_x, displacement, velocity, acceleration = run_jelly_simulation(cps_start, D, Re, U, ϵ, thk, deg, pathing)
plot_logger("test_psolver")
savefig("psolver.png")

open("results.csv", "w") do io
    println(io, "forces,time,time_sim,timesteps,displacement,velocity,acceleration")
    n = length(forces)
    for i in 1:n
        f       = forces[i]
        tnum    = time[i]
        tsim    = time_sim[i]
        tsteps  = timesteps[i]
        dis     = displacement[i]
        vel     = velocity[i]
        acc     = acceleration[i]
        println(io, "$f,$tnum,$tsim,$tsteps,$dis,$vel,$acc")
    end
end