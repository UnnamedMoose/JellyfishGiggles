"""
Function that is used to refine the polyline into multiple control points, that can be adjusted by the user. It is written so that the old control points will always be used in the new solution. A lower number of control points can therefore not be chosen. It is important to input the set of control points as a static matrix, otherwise it will give a type error.

    - `cpset::SMatrix{D,M,T}` = an individual control point set.
    - `N::Int` = Number of desired control points.

"""
function refine_polyline(cpset::SMatrix{D,M,T},N::Int) where {D,M,T}

    @assert M ≥ 2
    @assert N ≥ M "N must be ≥ number of control points"

    # Segment lengths
    seglen = [norm(cpset[:, i+1] - cpset[:, i]) for i in 1:M-1]
    L = sum(seglen)

    # Number of points per segment (excluding left endpoint)
    nseg = round.(Int, (seglen / L) .* (N - M))
    nseg[end] += (N - M) - sum(nseg)  # fix rounding error

    pts = Vector{SVector{D,T}}()
    push!(pts, cpset[:,1])  # first CP

    for i in 1:M-1
        p0 = cpset[:, i]
        p1 = cpset[:, i+1]

        for k in 1:nseg[i]
            α = k / (nseg[i] + 1)
            push!(pts, (1-α)*p0 + α*p1)
        end

        push!(pts, p1)  # preserve CP exactly
    end

    @assert length(pts) == N

    return SMatrix{D,N,T}(hcat(pts...))
end

"""
This is the optimiser function, it uses a reference value (area advised for 2D and volume for 3D). Input is one set of control points in the static matrix format. 

    -`cpset::SMatrix{2,N,T}` = input control point set, 1 frame.
    -`ref_val` = reference value, choose either the area or volume at `t=0`.
    -`ThreeD=false` for 2D simulations and `ThreeD=true` for 3D simulations.
"""
function optimiser(cpset::SMatrix{2,N,T}, ref_val; ThreeD=false) where {N,T}
    if ThreeD == true
        area = shape_volume(cpset)
    else
        area = shape_area(cpset)
    end
    s = √(ref_val / area)      
    cx = sum(cpset[1,i] for i in 1:N) / N
    cy = sum(cpset[2,i] for i in 1:N) / N
    cpsn = MMatrix{2,N,T}(cpset)

    for i in 1:N
        x = cpset[1,i]
        y = cpset[2,i]
        cpsn[1,i] = cx + s*(x - cx)
        cpsn[2,i] = cy + s*(y - cy)
    end

    return SMatrix{2,N,T}(cpsn)
end

"""
Forward and backward pass exponential smoothing function. The input is the array that should be smoothened with a smoothing factor. Note that this is NOT a static matrix input, but a pathing array for the case of control point smoothing, as it is a general exponential smoothing algorithm. The result is a smoothened array.

    - `x::Vector{Float64}` = Input array that requires smoothing.
    - `α::T` = Smoothing factor. Lower value equals more smoothing. Higher value reacts quicker to recent changes.
"""
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