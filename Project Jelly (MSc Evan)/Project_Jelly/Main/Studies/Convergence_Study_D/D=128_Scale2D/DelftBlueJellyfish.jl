using LinearAlgebra, Printf, Statistics, Plots, ParametricBodies, StaticArrays
using WaterLily, CUDA
using GeometryBasics, Optim
using BiotSavartBCs
using Interpolations: LinearInterpolation
using DelimitedFiles, DataFrames, CSV
using Dierckx

import WaterLily: @loop,scale_u!,conv_diff!,udf!,accelerate!,BDIM!

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

import BiotSavartBCs: biot_mom_step!,biot_project!
function biot_mom_step!(a::Flow{N},b,ω...;λ=quick,udf=nothing,fmm=true,U,kwargs...) where N
    a.u⁰ .= a.u; scale_u!(a,0); t₁ = sum(a.Δt); t₀ = t₁-a.Δt[end]
    # predictor u → u'
    @log "p"
    conv_diff!(a.f,a.u⁰,a.σ,λ,ν=a.ν)
    udf!(a,udf,t₀; kwargs...)
    BDIM!(a);
    biot_project!(a,b,ω...,U;fmm) # new
    # corrector u → u¹
    @log "c"
    conv_diff!(a.f,a.u,a.σ,λ,ν=a.ν)
    udf!(a,udf,t₁; kwargs...)
    BDIM!(a); scale_u!(a,0.5)
    biot_project!(a,b,ω...,U;fmm,w=0.5) # new
    push!(a.Δt,CFL(a))
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
    Npts    = size(cps_seq[1], 2)
    t       = 1:Nframes
    mid     = split === nothing ? (Nframes ÷ 2) : split

    t1 = range(first(t), mid; length = (mid - first(t)) * up1 + 1)
    t2 = range(mid, last(t);  length = (last(t) - mid) * up2 + 1)
    t_interp = [t1; t2[2:end]]  # drop duplicate 'mid'

    splx = Vector{Spline1D}(undef, Npts)
    sply = Vector{Spline1D}(undef, Npts)
    for i in 1:Npts
        xs = [cps_seq[k][1,i] for k in t]
        ys = [cps_seq[k][2,i] for k in t]
        splx[i] = Spline1D(t, xs; k=3, s=s)
        sply[i] = Spline1D(t, ys; k=3, s=s)
    end

    out = Vector{SMatrix{2, Npts, Float64, 2Npts}}(undef, length(t_interp))
    for (j, τ) in enumerate(t_interp)
        M = Matrix{Float64}(undef, 2, Npts)
        for i in 1:Npts
            M[1,i] = splx[i](τ)
            M[2,i] = sply[i](τ)
        end
        out[j] = SMatrix{2, Npts, Float64, 2Npts}(M)
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
        zcoords = zeros(1, Npoints)
        cps_xy = SMatrix{2,Npoints,Float64}(hcat([f(t) for f in interp_funcs]...) )
        cps_t  = SMatrix{3,Npoints,Float64}(vcat(cps_xy, zcoords))
    else
        cps_t = SMatrix{2,Npoints,Float64}(hcat([f(t) for f in interp_funcs]...) )
    end
    return cps_t
end

function blend_cycles(v::Vector{Float64}, n_cycles::Int)
    n = length(v)
    overlap = 10
    result = copy(v)

    for _ in 2:n_cycles
        a = v[end-overlap+1:end]
        b = v[1:overlap]
        blend = (1 .- range(0, 1, length=overlap)) .* a .+ range(0, 1, length=overlap) .* b
        result = vcat(result[1:end-overlap], blend, v[overlap+1:end])
    end
    return result
end

function construct_jelly_motion(Ncps::Int=50,spline_s::Float64=0.001, up::Int=15, deg::Any=2; γ::Int=2, λ_area::T=T(1e-1), λ_shape::T=T(1e-3), Dmax::T = T(1.25))
    cps_0   = SA{T}[0.000  0.024  0.064  0.175  0.323  0.481  0.643  0.798  0.996  1.191  1.228  1.211  1.171  1.178  1.154  0.804  0.606  0.475  0.397  0.340  0.323  
                0.000  0.193  0.333  0.475  0.578  0.623  0.627  0.614  0.571  0.524  0.484  0.213  0.216  0.412  0.456  0.451  0.420  0.348  0.266  0.151 0.000  ] ./ Dmax
    cps_1   = SA{T}[0.000  0.024  0.067  0.175  0.326  0.488  0.639  0.794  0.993  1.164  1.198  1.290  1.252  1.171  1.090  0.801  0.606  0.478  0.404  0.343  0.323  
                    0.000  0.193  0.326  0.469  0.564  0.609  0.606  0.590  0.537  0.501  0.470  0.215  0.213  0.412  0.426  0.438  0.396  0.335  0.256  0.154  0.000  ] ./ Dmax
    cps_2   = SA{T}[0.000  0.024  0.081  0.188  0.333  0.481  0.639  0.798  0.986  1.150  1.222  1.385  1.353  1.202  1.154  0.801  0.616  0.501  0.427  0.370  0.323  
                    0.000  0.193  0.319  0.455  0.547  0.589  0.583  0.550  0.473  0.400  0.350  0.185  0.160  0.306  0.334  0.380  0.366  0.311  0.243  0.148 0.000  ] ./ Dmax
    cps_3   = SA{T}[0.000  0.034  0.091  0.199  0.337  0.481  0.629  0.781  0.973  1.077  1.181  1.344  1.305  1.198  1.154  0.798  0.643  0.522  0.448  0.387  0.357  
                    0.000  0.193  0.322  0.448  0.541  0.568  0.566  0.533  0.449  0.412  0.392  0.145  0.132  0.293  0.301  0.343  0.342  0.287  0.226  0.141  0.000  ] ./ Dmax   
    cps_4   = SA{T}[0.000  0.027  0.081  0.199  0.357  0.478  0.626  0.781  0.976  1.178  1.222  1.228  1.183  1.180  1.151  0.801  0.633  0.525  0.438  0.384  0.347  
                    0.000  0.204  0.326  0.466  0.562  0.589  0.587  0.560  0.507  0.467  0.437  0.200  0.200  0.359  0.389  0.381  0.363  0.315  0.233  0.158  0.000  ] ./ Dmax
    cps_5   = SA{T}[0.000  0.027  0.074  0.182  0.340  0.471  0.629  0.791  0.989  1.185  1.222  1.158  1.120  1.160  1.127  0.798  0.616  0.498  0.407  0.360  0.337  
                    0.000  0.204  0.340  0.483  0.585  0.620  0.624  0.608  0.561  0.518  0.478  0.237  0.252  0.420  0.454  0.452  0.410  0.352  0.253  0.172  0.000  ] ./ Dmax
    cps_6   = SA{T}[0.000  0.027  0.074  0.182  0.330  0.464  0.629  0.794  0.989  1.185  1.225  1.158  1.120  1.159  1.141  0.794  0.613  0.488  0.394  0.343  0.330  
                    0.000  0.204  0.340  0.483  0.592  0.637  0.641  0.628  0.592  0.532  0.495  0.233  0.252  0.403  0.454  0.469  0.424  0.366  0.267  0.175  0.000  ] ./ Dmax
    cps_7   = SA{T}[0.000  0.027  0.074  0.182  0.330  0.464  0.629  0.794  0.989  1.185  1.225  1.180  1.144  1.164  1.141  0.794  0.613  0.488  0.394  0.343  0.330  
                    0.000  0.204  0.340  0.483  0.592  0.637  0.641  0.628  0.592  0.532  0.495  0.227  0.230  0.403  0.454  0.469  0.424  0.366  0.267  0.175  0.000  ] ./ Dmax
    cps_8   = SA{T}[0.000  0.027  0.074  0.182  0.330  0.464  0.629  0.794  0.989  1.185  1.225  1.187  1.151  1.164  1.141  0.794  0.613  0.488  0.394  0.343  0.330  
                    0.000  0.204  0.340  0.483  0.592  0.637  0.641  0.628  0.592  0.532  0.495  0.230  0.233  0.403  0.454  0.469  0.424  0.366  0.267  0.175  0.000  ] ./ Dmax
    cps_9   = SA{T}[0.000  0.027  0.074  0.182  0.330  0.464  0.629  0.794  0.989  1.185  1.225  1.193  1.154  1.164  1.141  0.794  0.613  0.488  0.394  0.343  0.330  
                    0.000  0.204  0.340  0.483  0.592  0.637  0.641  0.628  0.592  0.532  0.495  0.220  0.220  0.403  0.454  0.469  0.424  0.366  0.267  0.175  0.000  ] ./ Dmax
    start   = SA{T}[0.000 0.000 0.000 0.000; 0.000 -0.010 -0.020 -0.030]
    ending  = SA{T}[0.000 0.000 0.000 0.000; 0.030  0.020  0.010  0.000]

    cps_contr           = [cps_0, cps_1, cps_2, cps_3, cps_4, cps_5]
    cps_exp             = [cps_5, cps_6, cps_7, cps_8, cps_9, cps_0]

    cps_list_contr      = Vector{SMatrix{2,Ncps,Float64,2Ncps}}([resample_by_arclength(BSplineCurve(to_static(cps[:,1:end]); degree=deg), Ncps) for cps in cps_contr])
    cps_list_exp        = Vector{SMatrix{2,Ncps,Float64,2Ncps}}([resample_by_arclength(BSplineCurve(to_static(cps[:,1:end]); degree=deg), Ncps) for cps in cps_exp])   
    cps_contr           = Vector{SMatrix{2,Ncps,Float64,2Ncps}}(vcat(smooth_spline_variable_upsample(cps_list_contr; s=spline_s, up1=up, up2=up, split=nothing) ))
    cps_exp             = Vector{SMatrix{2,Ncps,Float64,2Ncps}}(vcat(smooth_spline_variable_upsample(cps_list_exp; s=spline_s, up1=Int(up*γ), up2=Int(up*γ), split=nothing) ))

    cps_contr           = cps_contr |> make_symmetric_jelly_new |> x -> [hcat(cp, cp[:, 1]) for cp in x] |> reverse_cps_list |>
                            x -> [hcat(start, cps[:, 2:end-1], ending) for (i, cps) in enumerate(x)] |>
                            x -> [SMatrix{2,2Ncps+5,Float64}(cps) for cps in x]

    cps_exp             = cps_exp |> make_symmetric_jelly_new |> x -> [hcat(cp, cp[:, 1]) for cp in x] |> reverse_cps_list |>
                            x -> [hcat(start, cps[:, 2:end-1], ending) for (i, cps) in enumerate(x)] |>
                            x -> [SMatrix{2,2Ncps+5,Float64}(cps) for cps in x]

    cps_list_new        = Vector{SMatrix{2,2Ncps+5,Float64}}(vcat(cps_contr, cps_exp))
    # cps_list_new        = Vector{SMatrix{2,2Ncps+5,Float64}}([optimize_control_points(cps, get_area(cps_list_new[1]); λ_area=λ_area, λ_shape=λ_shape, degree=deg) for cps in cps_list_new])

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

function gen_p_plots(sim, tᵢ, Domain)
    save_dir_p = joinpath("Prolate Jellyfish", "Pressure_check")
    isdir(save_dir_p) || mkpath(save_dir_p)

    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))

    p = Array(sim.flow.p)        
    σ = Array(sim.flow.σ)          

    p_masked = copy(p)              
    p_masked[σ .< 0] .= NaN

    Nx, Ny = size(p_masked)
    x = range(0, Domain; length = Nx) ./ sim.L
    y = range(0, Domain; length = Ny) ./ sim.L

    pressure_plot = Plots.heatmap(x, y, p_masked', aspect_ratio=1,
    xlims=(0, Domain/sim.L), ylims=(0, Domain/sim.L), c=:balance, clims=(-2, 2),
    xlabel="x", ylabel="y", title="Pressure Field")

    Plots.contour!(x,y,sim.flow.σ',levels=[0])   
    savefig(pressure_plot, joinpath(save_dir_p, "pressure_$(tᵢ).png"))
end

function gen_u_plots(sim, tᵢ, Domain)
    save_dir_p = joinpath("Prolate Jellyfish", "Velocity_check")
    isdir(save_dir_p) || mkpath(save_dir_p)

    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))

    u = Array(.√(sim.flow.u[:,:,1].^2 .+ sim.flow.u[:,:,2].^2))
    σ = Array(sim.flow.σ)    

    Nx, Ny = size(u)
    x = range(0, Domain; length = Nx) ./ sim.L
    y = range(0, Domain; length = Ny) ./ sim.L

    u_masked = copy(u)              
    u_masked[σ .< 0] .= NaN      

    pressure_plot = Plots.heatmap(x,y,u_masked', aspect_ratio=1,
    xlims=(0, Domain/sim.L), ylims=(0, Domain/sim.L), c=:balance, clims=(-2, 2),
    xlabel="x", ylabel="y", title="Velocity Field")

    Plots.contour!(x,y,sim.flow.σ',levels=[0])   
    savefig(pressure_plot, joinpath(save_dir_p, "velocity_$(tᵢ).png"))
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

function gen_ω_gif(sim, t, Domain)
    save_dir_ω = joinpath("Prolate Jellyfish", "Vorticity_check")
    isdir(save_dir_ω) || mkpath(save_dir_ω)
    @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L/sim.U
    @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I]) < 0.001, 0.0, sim.flow.σ[I])
    ω = Array(sim.flow.σ)

    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
    σ = Array(sim.flow.σ)
    ω_masked = copy(ω)
    ω_masked[σ .< 0] .= NaN

    vorticity_plot = WaterLily.flood(ω_masked,clims=(-5,5),
              cfill=:seismic,legend=false,border=:none, xlims=(0, Domain),ylims=(0, Domain),
              xlabel="x", ylabel="y", title="Vorticity at tU/D=$(round(t, digits=4))")

    vorticity_plot = Plots.contour!(sim.flow.σ',levels=[0])
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

knots_vector(p::Int, Ncp::Int) = vcat(zeros(p+1), (Ncp-p-1 > 0 ? collect(range(0.0, 1.0, length=Ncp-p+1))[2:end-1] : Float64[]), ones(p+1))
           
D           = 2^7     
Domain      = 6D          
Re          = 302               
St          = 0.52              
U           = 1                 
ϵ           = 1                 
thk         = 0            
deg         = 1                
cycles      = 35                 
period      = 1       
Ncps        = 50
γ           = 2
Uff         = 0

path_x, path_y                                      = construct_jelly_motion(50,0.001,10,deg;γ = γ)      # creates 152 frames per period
path_x                                              = [blend_cycles(p, 65) for p in path_x]
path_y                                              = [blend_cycles(p, 65) for p in path_y]
len                                                 = length(path_x)
path_x_smooth                                       = [exp_smooth(path_x[i], 0.25) for i in 1:len]
path_y_smooth                                       = [exp_smooth(path_y[i], 0.25) for i in 1:len]  
frame_points                                        = range(1,length(path_x_smooth[25]), step=1)
pathing                                             = control_point_functions(path_x_smooth, path_y_smooth, frame_points)

fall!(flow,t;acceleration) = for i ∈ 1:ndims(flow.p)
    WaterLily.@loop flow.f[I,i] += acceleration[i] over I ∈ CartesianIndices(flow.p)
end

function run_jelly_simulation(period, D, Re, U, ϵ, pathing, Domain, Uff, Ncps)
    cps         =   cps_at_time(pathing, 2*Ncps+5, 0;) .* 2D .+ SA{T}[2D, 2.5D] # defined from t = 0 to t = 545, which are actually frames.
    weights     =   ones(T, size(cps, 2)); knots       =   Float64.(knots_vector(deg, size(cps, 2))); curve       =   NurbsCurve(cps, knots, weights )
    body        =   DynamicNurbsBody(curve; thk=0, boundary=true)
    sim         =   BiotSimulation((Domain, Domain), (Uff,Uff), D; U, ν = U * D / Re, body, T, mem=Array, ϵ)
    forces      = []; force_drag = []; force_inertia = []; force_addedmass = []; time = []; displacement = []; velocity = []; acceleration = []
    duration    = 25; t₀ = round(sim_time(sim)); step = 0.1; t0 = 0; a0 = 0; v0 = 0; p0 = 0

    for tᵢ in range(t₀, t₀ + duration; step)        
        t = sum(sim.flow.Δt[1:end-1])
        while t < tᵢ * sim.L / sim.U
            cps             = cps_at_time(pathing, 2*Ncps+5, t*(142/(period*D))) .* 2D .+ SA{T}[2D, 2.5D]
            d = maximum(cps[2,:]) - 2.5D; h = maximum(cps[1,:]) - 2D; α = (2*h / d)^1.4
            sim.sim.body    = ParametricBodies.update!(sim.sim.body, cps, sim.flow.Δt[end])

            measure!(sim)
            biot_mom_step!(sim.flow,sim.pois,sim.ω,sim.x₀,sim.tar,sim.ftar;
                           fmm=sim.fmm,udf=fall!,acceleration=SA[-a0,0.f0],U=SA[-v0,0.0]) # change of frame

            force           =   -WaterLily.pressure_force(sim)[1]
            Δt              =   sim.flow.Δt[end]
            force_dr        =   24 / (Re^(0.7)) * 0.5 * get_area(cps) * v0
            accel           =   (force + α * get_area(cps) * a0) / (get_area(cps) * (1 + α))
            force_in        =   get_area(cps) * accel
            force_am        =   α * get_area(cps) * (accel - a0)

            p0              +=  Δt * (v0 + Δt * accel / 2.)
            v0              +=  Δt * accel
            a0              =   accel

            if force == NaN
                println("Diverging Solution")
            end

            if t > period * D
                push!(velocity, v0)
                push!(displacement, p0)
                push!(acceleration, a0)
                push!(force_addedmass, force_am)
                push!(force_inertia, force_in)
                push!(force_drag, force_dr)
                push!(time, t * sim.U / sim.L)
                push!(forces, force)
            end

            t0 = t; t += sim.flow.Δt[end]
        end

        gen_p_plots(sim, tᵢ, Domain)
        gen_u_plots(sim, tᵢ, Domain)
        gen_ω_gif(sim, tᵢ, Domain)
        
        println("tU/L=", round(tᵢ, digits = 4), ", Δt=", round(sim.flow.Δt[end], digits = 3))
    end 
    return forces, force_addedmass, force_inertia, force_drag, time, displacement, velocity, acceleration
end

WaterLily.logger("test_psolver")
forces, force_addedmass, force_inertia, force_drag, time, displacement, velocity, acceleration = run_jelly_simulation(period, D, Re, U, ϵ, pathing, Domain, Uff, Ncps)
plot_logger("test_psolver")
savefig("psolver.png")

open("results.csv", "w") do io
    println(io, "forces,force_addedmass,force_inertia,time,displacement,velocity,acceleration")
    n = length(forces)
    for i in 1:n
        ftot    = forces[i]
        fam     = force_addedmass[i]
        fin     = force_inertia[i]
        fdr     = force_drag[i]
        tnum    = time[i]
        dis     = displacement[i]
        vel     = velocity[i]
        acc     = acceleration[i]
        println(io, "$ftot,$fam,$fin,$fdr,$tnum,$dis,$vel,$acc")
    end
end