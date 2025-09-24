function create_cps_list(::Type{T}) where {T<:AbstractFloat}
    # t = 0
    cps_0 = SA{T}[0.000  0.024  0.064  0.175  0.323  0.481  0.643  0.798  0.996  1.191  1.228  1.201  1.171  1.178  1.154  0.804  0.606  0.475  0.397  0.340  0.323  
                    0.000  0.193  0.333  0.475  0.578  0.623  0.627  0.614  0.571  0.524  0.484  0.213  0.216  0.412  0.456  0.451  0.420  0.348  0.266  0.151 0.000  ] #*L .+ SA{T}[2L,3L]
    # t = T/10
    cps_1 = SA{T}[0.000  0.024  0.067  0.175  0.326  0.488  0.639  0.794  0.993  1.164  1.198  1.296  1.265  1.171  1.090  0.801  0.606  0.478  0.404  0.343  0.323  
                    0.000  0.193  0.326  0.469  0.564  0.609  0.606  0.590  0.537  0.501  0.470  0.216  0.206  0.412  0.426  0.438  0.396  0.335  0.256  0.154  0.000  ] # *L .+ SA{T}[2L, 3L]
    # t = 2T/10
    cps_2 = SA{T}[0.000  0.024  0.081  0.188  0.333  0.481  0.639  0.798  0.986  1.171  1.222  1.390  1.373  1.205  1.154  0.801  0.616  0.501  0.427  0.370  0.323  
                    0.000  0.193  0.319  0.455  0.547  0.589  0.583  0.550  0.473  0.419  0.321  0.173  0.159  0.304  0.334  0.380  0.366  0.311  0.243  0.148 0.000  ] #*L .+ SA{T}[2L,3L]
    # t = 3T/10
    cps_3 = SA{T}[0.000  0.034  0.091  0.199  0.337  0.481  0.629  0.781  0.973  1.077  1.181  1.346  1.319  1.198  1.154  0.798  0.643  0.522  0.448  0.387  0.357  
                    0.000  0.193  0.322  0.448  0.541  0.568  0.566  0.533  0.449  0.412  0.392  0.149  0.135  0.291  0.301  0.343  0.342  0.287  0.226  0.141  0.000  ] # *L .+ SA{T}[2L, 3L]    
    # t = 4T/10
    cps_4 = SA{T}[0.000  0.027  0.081  0.199  0.357  0.478  0.626  0.781  0.976  1.178  1.222  1.228  1.191  1.195  1.151  0.801  0.633  0.525  0.438  0.384  0.347  
                    0.000  0.204  0.326  0.466  0.562  0.589  0.587  0.560  0.507  0.467  0.437  0.200  0.206  0.359  0.389  0.381  0.363  0.315  0.233  0.158  0.000  ] # *L .+ SA{T}[2L, 3L]
    # t = 5T/10
    cps_5 = SA{T}[0.000  0.027  0.074  0.182  0.340  0.471  0.629  0.791  0.989  1.185  1.222  1.151  1.124  1.164  1.127  0.798  0.616  0.498  0.407  0.360  0.337  
                    0.000  0.204  0.340  0.483  0.585  0.620  0.624  0.608  0.561  0.518  0.478  0.237  0.250  0.420  0.454  0.452  0.410  0.352  0.253  0.172  0.000  ] # *L .+ SA{T}[2L, 3L]
    # t = 6T/10
    cps_6 = SA{T}[0.000  0.027  0.074  0.182  0.330  0.464  0.629  0.794  0.989  1.185  1.225  1.158  1.124  1.164  1.141  0.794  0.613  0.488  0.394  0.343  0.330  
                    0.000  0.204  0.340  0.483  0.592  0.637  0.641  0.628  0.592  0.532  0.495  0.233  0.250  0.403  0.454  0.469  0.424  0.366  0.267  0.175  0.000  ] # *L .+ SA{T}[2L, 3L]
    # t = 7T/10
    cps_7 = SA{T}[0.000  0.027  0.074  0.182  0.330  0.464  0.629  0.794  0.989  1.185  1.225  1.178  1.144  1.164  1.141  0.794  0.613  0.488  0.394  0.343  0.330  
                    0.000  0.204  0.340  0.483  0.592  0.637  0.641  0.628  0.592  0.532  0.495  0.227  0.230  0.403  0.454  0.469  0.424  0.366  0.267  0.175  0.000  ] # *L .+ SA{T}[2L, 3L]
    # t = 8T/10
    cps_8 = SA{T}[0.000  0.027  0.074  0.182  0.330  0.464  0.629  0.794  0.989  1.185  1.225  1.185  1.151  1.164  1.141  0.794  0.613  0.488  0.394  0.343  0.330  
                    0.000  0.204  0.340  0.483  0.592  0.637  0.641  0.628  0.592  0.532  0.495  0.230  0.233  0.403  0.454  0.469  0.424  0.366  0.267  0.175  0.000  ] # *L .+ SA{T}[2L, 3L]
    # t = 9T/10
    cps_9 = SA{T}[0.000  0.027  0.074  0.182  0.330  0.464  0.629  0.794  0.989  1.185  1.225  1.191  1.154  1.164  1.141  0.794  0.613  0.488  0.394  0.343  0.330  
                    0.000  0.204  0.340  0.483  0.592  0.637  0.641  0.628  0.592  0.532  0.495  0.220  0.220  0.403  0.454  0.469  0.424  0.366  0.267  0.175  0.000  ] # *L .+ SA{T}[2L, 3L]
    
    cps_list = [cps_0, cps_1, cps_2, cps_3, cps_4, cps_5, cps_6, cps_7, cps_8, cps_9, cps_0] #10-element (Vector{SMatrix{2, 22, Float32, 44}})

    # mirror_set(cps::SMatrix{2,N,T}) where {N,T} =                # Mirror around the x-axis: (x, y) -> (x, -y)
    #     SMatrix{2,2,T}(1, 0, 0, -1) * cps

    # mirror_cps_list(cps_list::AbstractVector{<:SMatrix{2,N,T}}) where {N,T} = # Creates a list with all mirrored control points of all sets
    #     map(mirror_set, cps_list)

    function make_symmetric_jelly(cps_list::AbstractVector{<:SMatrix{2,N,T}};
                                        tol = nothing) where {N,T}
        tol === nothing && (tol = sqrt(eps(T)))

        first_cps = cps_list[1]::SMatrix{2,21,Float32,42}
        keep_idxs = findall(j -> abs(first_cps[2, j]) > tol, 1:N)   # length K
        K = length(keep_idxs)
        M = N + K + 1                                               # Final column count, added 1 for the 0,0 point
        keep_rev = Tuple(reverse(keep_idxs))                        # Reverse order for mirrored control points

        build_sym_jelly(cps::SMatrix{2,N,T}) where {N,T} = SMatrix{2,M,T}(
            ntuple(2M) do k
                if k <= 2N                                          # Original control points   
                    i = (k - 1) % 2 + 1
                    j = (k + 1) ÷ 2
                    cps[i, j]
                elseif k <= 2N + 2K                                 # Create the mirrored control points and add them to the original
                    kk  = k - 2N
                    i   = (kk - 1) % 2 + 1
                    jth = (kk + 1) ÷ 2
                    col = keep_rev[jth]
                    i == 1 ? cps[1, col] : -cps[2, col]
                else
                    0                                               # Add the 0,0 point, 1st data point, again at the end.
                end
            end
        )

        map(build_sym_jelly, cps_list)                                    
    end

    reverse_cps_order(cps::SMatrix{2,N,T}) where {N,T} =        # To create the counterclockwise order
        SMatrix{2,N,T}(cps[:, reverse(1:N)])                        # Make SMatrix so that my NURBS function works

    reverse_cps_list(cps_list::AbstractVector{<:SMatrix{2,N,T}}) where {N,T} =
        map(reverse_cps_order, cps_list)

    # cps_list = mirror_cps_list(cps_list)                              # Mirror the shape around the x-axis
    # @show typeof(cps_list)
    
    cps_list = make_symmetric_jelly(cps_list)                   # Vector{Matrix{Float32}}
    @show typeof(cps_list)
    new_cps_list = reverse_cps_list(cps_list)                       # Change the cps_list from clockwise to counterclockwise order
return new_cps_list
end

# Initialize the B-spline curves and calculate areas
function optimize_control_points(cps::SMatrix{2,N,T}, reference_area; λ_area::T = T(2), λ_shape::T = T(1e-3), degree::Int = 2, nsamples::Int = 500) where {N,T}
    x0 = vec(Array{Float32}(cps))           # 2N-vector
    s_vals = range(0, stop=1, length=nsamples)

    cost = function (x::AbstractVector)
        X = reshape(x, 2, N)
        cps_new = SMatrix{2,N,T}(Tuple(vec(X))...)
        curve = BSplineCurve(cps_new; degree=degree)

        pts = (curve(s) for s in s_vals)
        pts_collected = collect(pts)

        area = poly_area(pts_collected)
        area_error = area - reference_area
        shape_error = sum(abs2, x .- x0)

        return (λ_area * area_error^2) + (λ_shape * shape_error)
    end

    res = optimize(cost, copy(x0), NelderMead())
    xopt = Optim.minimizer(res)
    Xopt = reshape(xopt, 2, N)

    return SMatrix{2,N,Float32}(Tuple(vec(Xopt))...)
end