using LinearAlgebra, Printf, Statistics, Plots, ParametricBodies, StaticArrays
using WaterLily, CUDA
using GeometryBasics, Optim
using BiotSavartBCs
using Interpolations: Flat, LinearInterpolation
using DelimitedFiles, DataFrames
using CairoMakie
using Dierckx

include("Metrics.jl")

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

    cps_list_og     = [cps_0, cps_1, cps_2, cps_3, cps_4, cps_5, cps_6, cps_7, cps_8, cps_9, cps_0] 

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

    start = SA{T}[0.000 0.000 0.000 0.000;
                0.000 -0.010 -0.020 -0.030]
    ending = SA{T}[0.000 0.000 0.000 0.000;
                0.030 0.020 0.010 0.000]
    curves          = [BSplineCurve(cps; degree=2) for cps in cps_list_og]
    Npoints         = 60
    cps_list_og     = [resample_by_arclength(curve, Npoints) for curve in curves]

    cps_list        = make_symmetric_jelly_new(cps_list_og)     
    cps_list        = [hcat(cps, cps[:, 1]) for cps in cps_list]                 

    new_cps_list    = reverse_cps_list(cps_list)                 
    new_cps_list    = [hcat(start, cps[:,2:end-1], ending) for (i,cps) in enumerate(new_cps_list)]
    new_cps_list    = [SMatrix{2, 125, Float64, 250}(cps) for cps in new_cps_list]        



    function smooth_spline_variable_upsample(cps_seq; s=0.001, up1=10, up2=5, split=nothing)
        Nframes = length(cps_seq)
        Npts    = size(cps_seq[1], 2)
        t       = 1:Nframes
        mid     = split === nothing ? (Nframes ÷ 2) : split

        t1 = range(first(t), mid; length = (mid - first(t)) * up1 + 1)
        t2 = range(mid, last(t);  length = (last(t) - mid) * up2 + 1)
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
            out[j] = SMatrix{2, Npts, Float64, 250}(M)
        end
        return out
    end

    dense_cps= smooth_spline_variable_upsample(new_cps_list; s=0.01, up1=10, up2=10)
    
    return dense_cps
end

@inline function interpolate_cps_hermite_new(new_cps_list, t::Tp, period; nphases::Int=10, tangent_scale=0.5) where Tp
    τ_total = t / period
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

    τ2 = τ_local^2
    τ3 = τ_local^3

    h00 = 2τ3 - 3τ2 + 1
    h10 = τ3 - 2τ2 + τ_local
    h01 = -2τ3 + 3τ2
    h11 = τ3 - τ2

    return h00 .* p0 .+ h10 .* m0 .+ h01 .* p1 .+ h11 .* m1
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
    forces_total = []


    anim = @animate for tᵢ in range(t₀, t₀+duration; step)
        while t < tᵢ * sim.L / sim.U
            sim.flow.Δt[end] = WaterLily.CFL(sim.flow; Δt_max=Tp(0.05))
            
            cps_interp= interpolate_cps_hermite_new(new_cps_list, t, period; nphases=101)
            body_interpolation = cps_interp .* 1 .* sim.L .+ (Tp(2sim.L), Tp(2.5sim.L))

            sim.sim.body = ParametricBodies.update!(sim.sim.body, body_interpolation, sim.flow.Δt[end])

            raw_total    = -WaterLily.total_force(sim)[1]
            push!(forces_total, raw_total)

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
        c=:balance,          
        clims=(-5,5),  
        title="Pressure Field")

        Plots.contour!(sim.flow.σ',levels=[-sim.ϵ,0,sim.ϵ])   
        savefig(pressure_plot, joinpath(save_dir_p, "pressure_$(tᵢ).png"))

        @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L/sim.U
        @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I]) < 0.001, 0.0, sim.flow.σ[I])
        flood(sim.flow.σ[R] |> Array; clims=(-0.5,0.5), kv...)

        plotbody && body_plot!(sim; title="$(round(t, digits=4))")

        verbose && println("t=", round(t, digits=4), ", Δt=", round(sim.flow.Δt[end], digits=5))
    end 

    gif(anim,"Swimming_Jelly.gif")

    return forces_total
end

new_cps_list   = cps_optimizer()

generate_sdf_plots(new_cps_list, thk, D, Tp, deg)           # Generate signed distance function input = (cps_list, thk, grid size, Type, poly degree)

D = 2^5; Re = 302; U = 1; ϵ = 1; thk = 2ϵ+√3; deg = 2; cycles = Tp(1); period = Tp(3); duration = cycles * period                                                     
sim             = TwoDimJellyfish(; new_cps_list, D, Re, U, ϵ, thk, deg, mem=Array, use_biotsavart=true) 

WaterLily.logger("test_psolver")
res             = simulate_Jelly!(sim, new_cps_list; duration, period, step = 0.1, remeasure = true, plotbody = true)
plot_logger("test_psolver")
savefig("psolver.png")



# open("results.csv", "w") do io
#     println(io, "force")  # optional header
#     for f in res
#         println(io, f)
#     end
# end