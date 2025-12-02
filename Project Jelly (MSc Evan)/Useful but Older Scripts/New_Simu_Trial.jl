using LinearAlgebra, Printf, Statistics, Plots, ParametricBodies, StaticArrays
using WaterLily, CUDA
using GeometryBasics, Optim
using BiotSavartBCs
using Interpolations: Flat, LinearInterpolation, CubicSplineInterpolation
using Interpolations
using DelimitedFiles, DataFrames, CSV
using GLMakie
using Dierckx

@info "Running with $(Threads.nthreads()) Julia threads"

Tp = Float64
T = Float64

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

function exp_smooth(x::Vector{T}, α::T) where {T<:Real}
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

function get_body!(bod,sim,t=WaterLily.time(sim))
    @inside sim.flow.σ[I] = WaterLily.sdf(sim.body,SVector(Tuple(I).-0.5f0),t)
    copyto!(bod,sim.flow.σ[inside(sim.flow.σ)])
end

addbody(x,y;c=:black) = Plots.plot!(Shape(x,y), c=c, legend=false)
function body_plot_new!(sim;levels=[0],lines=:black,R=inside(sim.flow.p),title)
    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
    Plots.contour!(sim.flow.σ[R]'|>Array;levels,lines, title=title) 
    Plots.plot!(sim.body, add_cp=:true)    
end

"""
The following function constructs the jellyfish motion, returning a list of control point matrices.
Ncps = number of control points to define half of the jellyfish.
spline_s = smoothness for defining the control points list based on the control point trajectories that are defined using a 1D Dierckx spline.
path_s = second smoothing parameter to define the smoothing for the exponential smoothing algorithm.
up = the number of samples between each initial frame.

T.B.A: Area conservation
"""

function construct_jelly_motion(Ncps=50,spline_s=0.001,path_s=0.75, up=15, cycles=5; ThreeD=false)
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
    cps_list_ini    = [cps_0, cps_1, cps_2, cps_3, cps_4, cps_5, cps_6, cps_7, cps_8, cps_9, cps_0, cps_0, cps_0, cps_0, cps_0, cps_0, cps_0, cps_0, cps_0]
    
    cps_contr       = [cps_0, cps_1, cps_2, cps_3, cps_4, cps_5]
    cps_exp        = [cps_5, cps_6, cps_7, cps_8, cps_9, cps_0]
    cps_list_contr  = Vector{SMatrix{2,50,Float64,100}}([resample_by_arclength(BSplineCurve(to_static(cps[:,1:end]); degree=2), Ncps) for cps in cps_contr])
    cps_list_exp   = Vector{SMatrix{2,50,Float64,100}}([resample_by_arclength(BSplineCurve(to_static(cps[:,1:end]); degree=2), Ncps) for cps in cps_exp])   
    cps_contr       = Vector{SMatrix{2,50,Float64,100}}(vcat([smooth_spline_variable_upsample(cps_list_contr; s=spline_s, up1=up*2, up2=up*2, split=nothing) for _ in 1:cycles]...))
    cps_exp        = Vector{SMatrix{2,50,Float64,100}}(vcat([smooth_spline_variable_upsample(cps_list_exp; s=spline_s, up1=up*4, up2=up*4, split=nothing) for _ in 1:cycles]...))
    cps_contr = cps_contr |> make_symmetric_jelly_new |> x -> [hcat(cp, cp[:, 1]) for cp in x] |> reverse_cps_list |>
    x -> [hcat(start, cps[:, 2:end-1], ending) for (i, cps) in enumerate(x)] |>
    x -> [SMatrix{2,105,Float64,210}(cps) for cps in x]
    cps_exp = cps_exp |> make_symmetric_jelly_new |> x -> [hcat(cp, cp[:, 1]) for cp in x] |> reverse_cps_list |>
    x -> [hcat(start, cps[:, 2:end-1], ending) for (i, cps) in enumerate(x)] |>
    x -> [SMatrix{2,105,Float64,210}(cps) for cps in x]
    cps_list_new = Vector{SMatrix{2,105,Float64,210}}(vcat(cps_contr, cps_exp))

    cps_list_og     = Vector{SMatrix{2,50,Float64,100}}([resample_by_arclength(BSplineCurve(to_static(cps[:,1:end]); degree=2), Ncps) for cps in cps_list_ini])
    cps             = vcat([smooth_spline_variable_upsample(cps_list_og; s=spline_s, up1=up*4, up2=up*2, split=nothing) for _ in 1:cycles]...)

    K               = length(cps)          
    M               = size(cps[1], 2)         
    cps_const       = [similar(cps[1]) for _ in 1:K] 
    trajectories    = []
    for col in 1:M
        traj         = resample_constant_speed([cps[k][:, col] for k in 1:K], K)
        push!(trajectories, traj)
        for k in 1:K
            cps_const[k][:, col] = traj[k]
        end
    end

    cps_const = Vector{SMatrix{2,50,Float64,100}}(cps)
    cps_const = cps_const |> make_symmetric_jelly_new |> x -> [hcat(cp, cp[:, 1]) for cp in x] |> reverse_cps_list |>
    x -> [hcat(start, cps[:, 2:end-1], ending) for (i, cps) in enumerate(x)] |>
    x -> [SMatrix{2,105,Float64,210}(cps) for cps in x]

    ThreeD && (new_cps_list = [SMatrix{3, size(cps,2)}(vcat(cps, zeros(1, size(cps,2)))) for cps in cps_const])

    # n_cps = length(cps_const[1][1,:]); cps_paths_x = [[] for _ in 1:n_cps]; cps_paths_y = [[] for _ in 1:n_cps]
    # for j in 1:length(cps_const)
    #     for (i, p) in enumerate(cps_const[j][1,:])
    #         push!(cps_paths_x[i], p)
    #     end
    #     for (i, p) in enumerate(cps_const[j][2,:])
    #         push!(cps_paths_y[i], p)
    #     end
    # end

    n_cps = length(cps_list_new[1][1,:]); cps_paths_x = [[] for _ in 1:n_cps]; cps_paths_y = [[] for _ in 1:n_cps]
    for j in 1:length(cps_list_new)
        for (i, p) in enumerate(cps_list_new[j][1,:])
            push!(cps_paths_x[i], p)
        end
        for (i, p) in enumerate(cps_list_new[j][2,:])
            push!(cps_paths_y[i], p)
        end
    end

    N = length(cps_const[1][1,:]); sx = Vector{Vector{T}}(undef, N); sy = Vector{Vector{T}}(undef, N); cps = Vector{SMatrix{2,N,T}}(undef, K)
    for i in 1:N
        sx[i] = exp_smooth(T.(cps_paths_x[i]), path_s)
        sy[i] = exp_smooth(T.(cps_paths_y[i]), path_s)
    end

    K = length(sx[1])
    for k in 1:K
        Ms = Matrix{Float64}(undef, 2, N)
        @inbounds for j in 1:N
            Ms[1,j] = sx[j][k]
            Ms[2,j] = sy[j][k]
        end
        cps[k] = SMatrix{2,N,Float64}(Ms)
    end

    ### Plotting routines for debugging
    # times = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    # plt = Plots.plot(new_cps_list[1][1,:], new_cps_list[1][2,:])
    # for i in times
    #     Plots.scatter!(new_cps_list[i][1,:], new_cps_list[i][2,:])
    # end
    # display(plt)
    # display(Plots.plot(cps_list[20][1,:], cps_list[20][2,:]))  
    # display(Plots.plot(new_cps_list[20][1,:], new_cps_list[20][2,:]))  
    display(Plots.plot(sx[25], xlabel="frame number", ylabel="Position", title="Position X CP25", label="cps_x 25"))
    display(Plots.plot(sy[25], xlabel="frame number", ylabel="Position", title="Position Y CP25", label="cps_y 25"))
    return cps_paths_x, cps_paths_y, cps_contr, cps_exp, cps_list_new
end

@inline function TwoDimJellyfish(::Type{T}=Float32; 
    new_cps_list, D=2^7, Re=302, U=1, ϵ=0.5, thk=2ϵ+√3, deg, 
    mem=Array, use_biotsavart=false) where {T<:AbstractFloat}

    cps = new_cps_list .* 1 .* D .+ SA{T}[2D, 2.5D]
    degree = deg
    n_ctrl = size(cps, 2)
    weights = ones(T, n_ctrl)
    knots = T.(clamped_uniform_knots(degree, n_ctrl))
    @show cps
    curve = NurbsCurve(cps, knots, weights)

    body = DynamicNurbsBody(curve; thk=thk, boundary=true)

    ν = U * D / Re

    return use_biotsavart ?
        BiotSimulation((6D, 6D), (0,0), D; U, ν, body, T, mem, ϵ) :
        Simulation((6D, 6D), (0,0), D; U, ν, body, T, mem, ϵ)
end

function get_forces!(sim, tᵢ, duration, new_cps_list;ThreeD=false)
    force = 0.0
    forces = Float64[]

    nphases = length(new_cps_list)
    t = sum(sim.flow.Δt[1:end-1])
    
    while t < tᵢ * sim.L / sim.U
        # sim.flow.Δt[end] = Tp(0.1)
        τ = t * sim.U / sim.L
        k = floor(Int, τ / duration * nphases)
        idx = mod(k, nphases) + 1
        if ThreeD
            body_interpolation = new_cps_list[idx] .* sim.L/2
        else
            body_interpolation = new_cps_list[idx] .* sim.L .+ (Tp(2sim.L), Tp(2.5sim.L))
        end
        sim.sim.body = ParametricBodies.update!(sim.sim.body, body_interpolation, sim.flow.Δt[end])
        # measure!(sim,t)
        sim_step!(sim, tᵢ; remeasure=true)
        # mom_step!(sim.flow, sim.pois)

        # sim_step!(sim, tᵢ; remeasure=true)
        force = -WaterLily.total_force(sim)[1] 
        @show idx, force
        push!(forces, force)
        t += sim.flow.Δt[end]
    end
    return force
end

# function get_forces!(sim, tᵢ, duration, new_cps_list;ThreeD=false)
#     force = 0.0
#     v = 0.0
#     s = 0.0
#     a = 0.0
#     vel = Float64[]
#     acc = Float64[]
#     pos = Float64[]
#     forces = Float64[]

#     nphases = length(new_cps_list)
#     Area = get_area(new_cps_list[1] .* sim.L)
#     t = sum(sim.flow.Δt[1:end-1])
#     while t < tᵢ * sim.L
#         sim.flow.Δt[end] = Tp(0.1)
#         k = floor(Int, t/(duration*sim.L) * nphases) ; idx0 = mod(k, nphases) + 1 #; idx1 = mod(k + 1, nphases) + 1
        
#         if ThreeD
#             body_interpolation = new_cps_list[idx0] .* sim.L/2
#         else
#             body_interpolation = new_cps_list[idx0] .* sim.L .+ (Tp(2sim.L), Tp(2.5sim.L))
#         end
#         @show typeof(body_interpolation)
#         sim.sim.body = ParametricBodies.update!(sim.sim.body, body_interpolation, sim.flow.Δt[end])
#         measure!(sim,t)
#         # sim_step!(sim, tᵢ; remeasure=true)
#         mom_step!(sim.flow, sim.pois)

#         # sim_step!(sim, tᵢ; remeasure=true)
#         force = -WaterLily.total_force(sim)[1] / (0.5*sim.L)
#         a = (force[1])/(Area)
#         v += sim.flow.Δt[end]*a
#         s += sim.flow.Δt[end]*(v+sim.flow.Δt[end]*a/2.)
#         @show idx0, force
#         push!(acc, a)
#         push!(vel, v)
#         push!(pos, s)
#         push!(forces, force)
#         t += sim.flow.Δt[end]
#     end
#     return force, v, s, a
# end

function gen_p_plots(sim, tᵢ)
    save_dir_p = joinpath(pwd(), "Pressure_check")
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

function gen_ω_gif(sim, t, R)
    @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L/sim.U
    @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I]) < 0.001, 0.0, sim.flow.σ[I])
    flood(sim.flow.σ[R] |> Array; clims=(-1, 1))
    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
    Plots.contour!(sim.flow.σ',levels=[-sim.ϵ,0,sim.ϵ], title="$(round(t, digits=4))")
end

function gen_div_gif(sim, t, R)
    @inside sim.flow.σ[I] = WaterLily.div(I,sim.flow.u)
    @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I]) < 0.001, 0.0, sim.flow.σ[I])
    flood(sim.flow.σ[R] |> Array; clims=(-0.5, 0.5))
    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
    Plots.contour!(sim.flow.σ',levels=[-sim.ϵ,0,sim.ϵ], title="$(round(t, digits=4))")
end

function make_circle_timed_sin_motion(T::Type, npoints::Int, t; radius=1.0, freq=0.1, amplitude=(1.0, 1.0))
    # θ = range(0, 2π, length=npoints)

    # shift = (amplitude[1] * sin(2π * freq * t), amplitude[2] * cos(2π * freq * t))
    # x = radius .* cos.(θ) .+ shift[1]
    # y = radius .* sin.(θ) .+ shift[2]
    # cps_list = SMatrix{2, npoints, T}(vcat(x', y')...)
    θ = range(0, 2π, length=npoints)
    x = 0.25 .* cos.(θ) .+ 0.25 * sin(2π * 0.01 * t)
    y = 0.25 .* sin.(θ) .+ 0.25 * cos(2π * 0.01 * t)
    cps_list = SMatrix{2, npoints, T}(vcat(x', y')...)
    return cps_list
end

function simulate_Jelly!(sim, new_cps_list; ThreeD=false,
                         duration=1, period=3, step=0.1, verbose=true,
                         R=inside(sim.flow.p), remeasure=false, plotbody=false, kv...)

    Tp = eltype(sim.flow.p)

    #period = period * sim.L / sim.U
    # nphases = length(new_cps_list)

    n_cps = length(new_cps_list)
    cps_paths_x = [[] for _ in 1:n_cps]  # vector of vectors
    cps_paths_y = [[] for _ in 1:n_cps]
    time_points = Float64[]
    forces = Float64[]
    time = Float64[]
    indices = []

    # steps = 3 * duration / nphases
    t₀ = sim_time(sim)
    path_x, path_y    = construct_jelly_motion(50,0.001,0.75,12,cycles; ThreeD=ThreeD)
    # t_points = range(1,length(path_x[1]) * (duration * sim.L / sim.U)/length(path_x[1]), step=1* (duration * sim.L / sim.U)/length(path_x[1]) )
    t_points = range(1,length(path_x[1]);step=1)
    pathing = control_point_functions(path_x, path_y, t_points)

    anim = @animate for tᵢ in range(t₀, t₀ + duration; step=step)
        t = sum(sim.flow.Δt[1:end-1])
        while t < tᵢ * sim.L / sim.U
            # τ = t * sim.U / sim.L
            # phase = τ / duration * nphases   # continuous phase in [0, nphases)

            # i = mod(floor(Int, phase), nphases) + 1
            # j = mod(i, nphases) + 1
            # w = phase - floor(phase)
            @show t
            # cps_interp = cps_at_time(pathing, 105, t)
            # # cps_interp = (1 - w) * new_cps_list[i] + w * new_cps_list[j]   # continuous
            # if ThreeD
            #     body_interpolation = cps_interp .* sim.L
            # else
            #     body_interpolation = cps_interp .* sim.L .+ (Tp(2sim.L), Tp(2.5sim.L))
            # end
            # sim.sim.body = ParametricBodies.update!(sim.sim.body, body_interpolation, sim.flow.Δt[end])
            measure!(sim,t) 
            mom_step!(sim.flow,sim.pois) # evolve Flow
            # sim_step!(sim, tᵢ; remeasure=true)
            @show sim_time(sim)
            for (i, p) in enumerate(cps_interp[1,:])
                push!(cps_paths_x[i], p)
            end
            # for (i, p) in enumerate(cps_interp[2,:])
            #     push!(cps_paths_y[i], p)
            # end
            push!(time_points, t)
            push!(time, sim_time(sim))

            t += sim.flow.Δt[end]
        end

        force = -WaterLily.total_force(sim)[1] / (0.5*sim.L)
        push!(forces, force)

        gen_p_plots(sim, t)
        gen_ω_gif(sim, t, R, kv...)
        # gen_div_gif(sim, t, R, kv...)

        verbose && println("t=", round(t, digits=4), ", Δt=", round(sim.flow.Δt[end], digits=5))
    end 

    gif(anim,"Swimming_Jelly.gif")

    return (forces=forces, cps_paths_x=cps_paths_x, cps_paths_y=cps_paths_y, time_num=time_points, time_sim=time, indices=indices)
end

# ThreeD = false
# new_cps_list = construct_jelly_motion(50,0.001,0.75,12,5; ThreeD=ThreeD)
# D = 2^5; Re = 302; U = 1; ϵ = 0.75; thk = 2ϵ+√3; deg = 2; cycles = Tp(5); period = Tp(1); duration = cycles * period  
# sim             = TwoDimJellyfish(; new_cps_list, D, Re, U, ϵ, thk, deg, mem=Array, use_biotsavart=true)
# WaterLily.logger("test_psolver")
# res = simulate_Jelly!(sim, new_cps_list; duration=1, period=period, step=0.1, remeasure=true, plotbody=false, ThreeD=ThreeD)
# plot_logger("test_psolver")
# savefig("psolver.png")

# open("results.csv", "w") do io
#     println(io, "time_num,time_sim,force")
#     n = length(res.forces)
#     for i in 1:n
#         f   = res.forces[i]
#         tnum = res.time_num[i]
#         tsim = res.time_sim[i]
#         println(io, "$tnum,$tsim,$f")
#     end
# end