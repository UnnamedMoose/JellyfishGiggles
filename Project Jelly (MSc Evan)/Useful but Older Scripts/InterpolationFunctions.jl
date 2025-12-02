Tp = Float32

"""
Fit and evaluate band-limited Fourier model for a set of CP time series.
- cps_series: Vector of SMatrix{2,N,T} at times t (length M)
- t: times (length M)
- t_eval: time(s) where you want CPs
- period: cycle period
- K: number of harmonics
Returns: SMatrix{2,N,Float64} (or Vector thereof if t_eval is a vector)
"""
function cps_fourier_interpolator(cps_series::Vector{<:SMatrix{2,N,T}}, t::AbstractVector,
                                  t_eval, period::Real, K::Int) where {N,T}
    M = length(t)
    @assert length(cps_series) == M
    # phase in [0, 2π)
    φ = 2π .* (t .% period) ./ period

    # Build design matrix Φ: [1, cosφ, sinφ, ..., cosKφ, sinKφ]
    function design(φ)
        Φ = Matrix{Float64}(undef, length(φ), 1 + 2K)
        Φ[:,1] .= 1.0
        for k in 1:K
            Φ[:, 2k]   = cos.(k .* φ)
            Φ[:, 2k+1] = sin.(k .* φ)
        end
        Φ
    end
    Φ = design(φ)

    # Stack CP coords over time into (M × N) matrices for x and y
    X = hcat([Array(cs)[1, :] for cs in cps_series]...)'  # M×N
    Y = hcat([Array(cs)[2, :] for cs in cps_series]...)'  # M×N

    # Solve least squares for each column j (independent CPs)
    # Coeff matrices: Cx, Cy are (1+2K) × N
    Cx = Φ \ X   # (1+2K)×N
    Cy = Φ \ Y

    # Evaluation helper
    function eval_at_time(tq)
        φq = 2π * (mod(tq, period) / period)
        Φq = design([φq])             # 1×(1+2K)
        xq = (Φq * Cx) |> vec         # length N
        yq = (Φq * Cy) |> vec
        # @SMatrix [xq'; yq']
        SMatrix{2,N,Float64}(hcat(xq, yq)'...)
    end

    if isa(t_eval, AbstractVector)
        return [eval_at_time(tq) for tq in t_eval]
    else
        return eval_at_time(t_eval)
    end
end


""" Interpolation functions for control point sets (CPS) using Hermite splines version 1. """
@inline function interpolate_cps_hermite(new_cps_list, t::Tp, Δt::Tp, sim, v::Tp, s::Tp, force; nphases::Int=10) where Tp
    period = Tp(6) * sim.L / sim.U
    τ_total = t / period

    k = floor(Int, τ_total * nphases)
    τ_local = τ_total * nphases - k

    idx0 = mod(k, nphases) + 1
    idx1 = mod(k + 1, nphases) + 1
    idx_prev = mod(k - 1, nphases) + 1
    idx_next = mod(k + 2, nphases) + 1
    interp_state =  τ_local + (idx0)
    p0 = new_cps_list[idx0]
    p1 = new_cps_list[idx1]
    m0 = (p1 - new_cps_list[idx_prev]) ./ 2
    m1 = (new_cps_list[idx_next] - p0) ./ 2

    τ = τ_local
    τ2 = τ^2
    τ3 = τ^3

    h00 = 2τ3 - 3τ2 + 1
    h10 = τ3 - 2τ2 + τ
    h01 = -2τ3 + 3τ2
    h11 = τ3 - τ2

    interpolated = (h00 .* p0 .+ h10 .* m0 .+ h01 .* p1 .+ h11 .* m1) #.* sim.L 

    if τ_total > 1
        a = (force / period) / (get_area(interpolated))
    else
        a = zero(Tp)
    end
    
    if 0 < idx0 < 4 
        v = Float32(v + a * Δt) 
    else 
        v = Float32(0)
    end

    s = Float32(1*(s + v * Δt))
    # @show s

    # offset = SVector{2,Tp}((4)*sim.L, 3sim.L)
    # interpolated = interpolated .+ offset

    area = get_area(interpolated ./ sim.L)
    push!(areas, area)
    push!(τ_locals, τ_local)

    return interpolated, v, s, interp_state
end

""" Interpolation functions for control point sets (CPS) using Hermite splines version 2. """
@inline function interpolate_cps_hermite_new(new_cps_list, t::Tp, period; nphases::Int=10, tangent_scale=0.5) where Tp
    τ_total = t / period
    k = floor(Int, τ_total * nphases)
    τ_local = τ_total * nphases - k
    
    idx0 = mod(k, nphases) + 1
    idx1 = mod(k + 1, nphases) + 1
    idx_prev = mod(k - 1, nphases) + 1
    idx_next = mod(k + 2, nphases) + 1

    cps_prev = new_cps_list[idx_prev]
    cps0     = new_cps_list[idx0]
    cps1     = new_cps_list[idx1]
    cps_next = new_cps_list[idx_next]

    # Damped Catmull-Rom style tangents
    max_speed = 0.02
    m0 = clamp.(tangent_scale .* (cps1 - cps_prev), -max_speed, max_speed)
    m1 = clamp.(tangent_scale .* (cps_next - cps0), -max_speed, max_speed)

    τ2 = τ_local^2
    τ3 = τ_local^3

    h00 = 2τ3 - 3τ2 + 1
    h10 = τ3 - 2τ2 + τ_local
    h01 = -2τ3 + 3τ2
    h11 = τ3 - τ2

    return h00 .* p0 .+ h10 .* m0 .+ h01 .* p1 .+ h11 .* m1
    # interpolated = (1-τ_local) .* p0 .+ τ_local .* p1 #.+ SA{Tp}[D,2D] # Linear fallback 
    return interpolated
end



"""
    build_cps_fourier(cps_series, t_series, period; K=4)

Fit a band-limited Fourier model to periodic control-point sets.

# Arguments
- cps_series :: Vector{SMatrix{2, N, T}}  → keyframe control-point sets
- t_series   :: AbstractVector{<:Real}    → times of each keyframe (same length)
- period     :: Real                      → total period of motion (in seconds or nondim)
- K          :: Int                       → number of Fourier harmonics (default 4)

# Returns
CPSFourier model (contains Fourier coefficients and metadata)
"""
function build_cps_fourier(cps_series,
                           t_series::AbstractVector{<:Real},
                           period::Real; K::Int=4) where {N,T}
    M = length(t_series)
    @assert M == length(cps_series) "Keyframe times and cps count must match."

    φ = 2π .* (t_series .% period) ./ period

    # Design matrix Φ: [1, cosφ, sinφ, ..., cosKφ, sinKφ]
    Φ = ones(M, 1 + 2K)
    for k in 1:K
        Φ[:, 2k]   .= cos.(k .* φ)
        Φ[:, 2k+1] .= sin.(k .* φ)
    end

    # Stack x, y coordinates over time
    X = reduce(vcat, [Float64.(Array(C)[1, :])' for C in cps_series])  # M×N
    Y = reduce(vcat, [Float64.(Array(C)[2, :])' for C in cps_series])  # M×N

    # Least-squares Fourier coefficients
    Cx = Φ \ X    # (1+2K) × N
    Cy = Φ \ Y

    return (; Cx, Cy, K, period, T=Float64)
end

"""
    eval_cps(model, t)

Evaluate control-point position, velocity, and acceleration at time `t`
from a Fourier model built by `build_cps_fourier`.

Returns three SMatrix{2,N,Float64} objects: (cps, dcps, acps)
"""
function eval_cps(model::NamedTuple, t::Real)
    K = model.K
    ω = 2π / model.period
    φ = ω * (t % model.period)

    # Preallocate basis vectors
    Φ  = Vector{Float64}(undef, 1 + 2K)
    dΦ = Vector{Float64}(undef, 1 + 2K)
    aΦ = Vector{Float64}(undef, 1 + 2K)
    Φ[1] = 1; dΦ[1] = 0; aΦ[1] = 0
    for k in 1:K
        c, s = cos(k*φ), sin(k*φ)
        Φ[2k]   = c;   Φ[2k+1] = s
        dΦ[2k]  = -k*ω*s;  dΦ[2k+1] =  k*ω*c
        aΦ[2k]  = -(k*ω)^2*c;  aΦ[2k+1] = -(k*ω)^2*s
    end

    # Evaluate Fourier sums
    x  = (Φ'  * model.Cx) |> vec
    y  = (Φ'  * model.Cy) |> vec
    vx = (dΦ' * model.Cx) |> vec
    vy = (dΦ' * model.Cy) |> vec
    ax = (aΦ' * model.Cx) |> vec
    ay = (aΦ' * model.Cy) |> vec

    N = length(x)
    T = Float64

    cps  = SMatrix{2, N, T}(hcat(x,  y )'...)
    dcps = SMatrix{2, N, T}(hcat(vx, vy)'...)
    acps = SMatrix{2, N, T}(hcat(ax, ay)'...)

    return cps, dcps, acps
end




# --- Geometry helpers (2D) ---
@inline cross2(a, b) = a[1]*b[2] - a[2]*b[1]
@inline function orientation(a, b, c)
    cross2(b .- a, c .- a)
end

@inline function on_segment(a, b, c; eps=1e-12)
    # c on segment ab (with small tolerance)
    min(a[1], b[1]) - eps ≤ c[1] ≤ max(a[1], b[1]) + eps &&
    min(a[2], b[2]) - eps ≤ c[2] ≤ max(a[2], b[2]) + eps &&
    abs(orientation(a, b, c)) ≤ eps
end

function segments_intersect(p1, p2, q1, q2; eps=1e-12)
    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    # Proper intersection
    if (o1*o2 < 0) && (o3*o4 < 0)
        return true
    end
    # Colinear / touching cases
    if abs(o1) ≤ eps && on_segment(p1, p2, q1; eps=eps); return true; end
    if abs(o2) ≤ eps && on_segment(p1, p2, q2; eps=eps); return true; end
    if abs(o3) ≤ eps && on_segment(q1, q2, p1; eps=eps); return true; end
    if abs(o4) ≤ eps && on_segment(q1, q2, p2; eps=eps); return true; end
    return false
end

"Return P' without duplicated last=first for closed curves."
function canonicalize_closed(P; eps=1e-12)
    N = size(P, 2)
    if N ≥ 2 && maximum(abs.(P[:, 1] .- P[:, end])) ≤ eps
        return P[:, 1:end-1]  # drop duplicate last point
    else
        return P
    end
end

@inline is_degenerate_edge(a, b; eps=1e-12) = 
    (abs(a[1]-b[1]) ≤ eps) && (abs(a[2]-b[2]) ≤ eps)

"""
    polyline_self_intersects(P; closed=true, eps=1e-12)

Return true if the polyline/polygon defined by 2×N matrix P self-intersects.
`closed=true` treats edge N→1 as an edge. Skips adjacent edges.
"""
function polyline_self_intersects(P; closed::Bool=true, eps::Real=1e-12)
    # Ensure unique vertices for closed curves (avoid zero-length N→1 edge)
    P2 = closed ? canonicalize_closed(P; eps=eps) : P

    N = size(P2, 2)
    if N < 4
        return false
    end
    last_edge = closed ? N : N - 1

    @inbounds for i in 1:last_edge
        i2 = (i == N) ? 1 : (i + 1)
        a1 = P2[:, i]; a2 = P2[:, i2]
        if is_degenerate_edge(a1, a2; eps=eps); continue; end

        for j in (i+1):last_edge
            j2 = (j == N) ? 1 : (j + 1)

            # skip edges sharing a vertex
            if i==j || i==j2 || i2==j || i2==j2; continue; end

            b1 = P2[:, j]; b2 = P2[:, j2]
            if is_degenerate_edge(b1, b2; eps=eps); continue; end

            if segments_intersect(a1, a2, b1, b2; eps=eps)
                return true
            end
        end
    end
    return false
end

"""
    is_simple_polygon(P; closed=true)

P is 2×N (columns are points). Returns true if polyline/polygon is simple.
"""
function is_simple_polygon(P; closed::Bool=true)
    N = size(P, 2)
    if N < 4; return true; end
    last_edge = closed ? N : N - 1
    for i in 1:last_edge
        i2 = (i % N) + 1
        for j in (i+1):last_edge
            j2 = (j % N) + 1
            # Skip edges that share a vertex
            if i==j || i==j2 || i2==j || i2==j2; continue; end
            if segments_intersect(P[:, i], P[:, i2], P[:, j], P[:, j2])
                return false
            end
        end
    end
    return true
end


@inline function interpolate_cps_hermite_safe(new_cps_list, t::Tp, period;
        nphases::Int=10, closed::Bool=true, max_iter::Int=25) where Tp

    τ_total = t / period
    k = floor(Int, τ_total * nphases)
    τ = τ_total * nphases - k

    idx0   = mod(k,     nphases) + 1
    idx1   = mod(k + 1, nphases) + 1
    idxprv = mod(k - 1, nphases) + 1
    idxnxt = mod(k + 2, nphases) + 1

    p0 = new_cps_list[idx0]   # 2×N
    p1 = new_cps_list[idx1]
    m0 = (new_cps_list[idx1]  - new_cps_list[idxprv]) ./ 2
    m1 = (new_cps_list[idxnxt] - new_cps_list[idx0])  ./ 2

    τ2 = τ^2; τ3 = τ^3
    h00 =  2τ3 - 3τ2 + 1
    h10 =  τ3 - 2τ2 + τ
    h01 = -2τ3 + 3τ2
    h11 =  τ3 - τ2

    H = h00 .* p0 .+ h10 .* m0 .+ h01 .* p1 .+ h11 .* m1   # Hermite
    L = (1 - τ) .* p0 .+ τ .* p1                            # Linear

    # If linear already self-intersects (rare if keyframes are simple and close),
    # fall back to the closer endpoint to keep things defined.
    if !is_simple_polygon(L; closed=closed)
        return (τ < 0.5 ? p0 : p1), k, τ_total, 0.0, true
    end

    # Accept Hermite if safe
    if is_simple_polygon(H; closed=closed)
        return H, k, τ_total, 1.0, false
    end

    # Binary search the largest α ∈ [0,1] with a simple curve
    lo, hi = 0.0, 1.0
    bestα  = 0.0
    for _ in 1:max_iter
        α = (lo + hi) / 2
        P = L .+ α .* (H .- L)
        if is_simple_polygon(P; closed=closed)
            bestα = α
            lo = α
        else
            hi = α
        end
    end
    Pbest = L .+ bestα .* (H .- L)
    return Pbest, k, τ_total   # last flag says "was clamped"
end
