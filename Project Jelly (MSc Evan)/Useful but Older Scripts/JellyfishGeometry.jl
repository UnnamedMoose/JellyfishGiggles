include("New_Geometry_Trial.jl")

function create_cps_list(::Type{T}) where {T<:AbstractFloat}
    """
    Initialize control points for the jellyfish geometry over time, derived from Sahin 2009. 
    All control points are entered as static arrays of type Float32, required for implementing ParametricBodies.jl functionalities.
    Size of the initial control point sets is (2,22).
    A start and ending part is added to acquire a C2 continuous closed curve.
    """
    # t = 0
    cps_0 = SA{T}[0.000  0.024  0.064  0.175  0.323  0.481  0.643  0.798  0.996  1.191  1.228  1.201  1.171  1.178  1.154  0.804  0.606  0.475  0.397  0.340  0.323  
                    0.000  0.193  0.333  0.475  0.578  0.623  0.627  0.614  0.571  0.524  0.484  0.213  0.216  0.412  0.456  0.451  0.420  0.348  0.266  0.151 0.000  ] #*L .+ SA{T}[2L,3L]
    # t = T/10
    cps_1 = SA{T}[0.000  0.024  0.067  0.175  0.326  0.488  0.639  0.794  0.993  1.164  1.198  1.296  1.265  1.171  1.090  0.801  0.606  0.478  0.404  0.343  0.323  
                    0.000  0.193  0.326  0.469  0.564  0.609  0.606  0.590  0.537  0.501  0.470  0.216  0.206  0.412  0.426  0.438  0.396  0.335  0.256  0.154  0.000  ] # *L .+ SA{T}[2L, 3L]
    # t = 2T/10
    cps_2 = SA{T}[0.000  0.024  0.081  0.188  0.333  0.481  0.639  0.798  0.986  1.150  1.222  1.390  1.373  1.205  1.154  0.801  0.616  0.501  0.427  0.370  0.323  
                    0.000  0.193  0.319  0.455  0.547  0.589  0.583  0.550  0.473  0.400  0.350  0.173  0.159  0.304  0.334  0.380  0.366  0.311  0.243  0.148 0.000  ] #*L .+ SA{T}[2L,3L]
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

    start = SA{T}[0.000 0.000 0.000 0.000;
            0.000 -0.010 -0.020 -0.030]
    ending = SA{T}[0.000 0.000 0.000 0.000;
                0.030 0.020 0.010 0.000]
    """
    Resample a parametric curve into an SMatrix{2,N,T}.
    curve(s) return an SVector{2,T}.
    """
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
    """
    Optimisation of the control points through area conservation and shape preservation, λ values can be adjusted in order to prioritise one over the other.
    The first and last control points are fixed in space, only the inner control points are optimised.

    Area is evaluated by sampling 500 points along the curves and calculating the polygon area.
    Shape is preserved by minimising the squared distance between the original and new inner control points.

    T.B.A. Not change the flap coordinates, only the bell coordinates.
    """
    function optimize_control_points(cps::SMatrix{2,N,T}, reference_area;
                                    λ_area::T = T(1e-2),
                                    λ_shape::T = T(1e-3),
                                    degree::Int = 2,
                                    nsamples::Int = 500) where {N,T}

        fixed_pt = cps[:, 1]  # first and last point (fixed)
        last_pt = cps[:, 11:end]
        fixed_pts = cps[:, ]
        # n_inner = N - 2       # number of inner points
        
        # Vectorize inner control points only
        # x0_inner = vec(Matrix(cps[:, 2:N-1]))  # 2*(N-2) vector
        x0_inner =  vec(Matrix(cps[:, 2:10]))
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

        # Rebuild full matrix
        Xopt_full = hcat(fixed_pt, Xopt_inner, last_pt)
        return SMatrix{2,N,T}(Tuple(vec(Xopt_full))...)
    end           

    """
    Function to make the jellyfish geometry symmetric about the x-axis.
    This is done by mirroring the control points with positive y-values to the negative side. 
    The mirrored control points are added to the original set, and a (0,0) point is added at the end to close the shape.
    A tolerance can be set to ignore very small y-values close to zero, which may arise due to numerical inaccuracies.
    Finally, the control points are reversed to make them order counterclockwise, as this will result in a closed body in ParametricBodies.jl.
    """

    function make_symmetric_jelly(cps_list::AbstractVector{<:SMatrix{2,N,T}};
                                        tol = nothing) where {N,T}
        tol === nothing && (tol = sqrt(eps(T)))

        first_cps = cps_list[1]::SMatrix{2,N,Float32,2N}
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
    

    cps_list_og     = [cps_0, cps_1, cps_2, cps_3, cps_4, cps_5, cps_6, cps_7, cps_8, cps_9] #10-element (Vector{SMatrix{2, 22, Float32, 44}})
    # cps_list_cleaned = [cps[:, [1:9; 11:13; 15:end]] for cps in cps_list_og]
    # cps_list_og = [SMatrix{2, 19, T}(cps) for cps in cps_list_cleaned]
    curves          = [BSplineCurve(cps; degree=2) for cps in cps_list_og]
    Npoints         = 50
    cps_list_og     = [resample_curve(curve, Npoints) for curve in curves]
    # Control point optimisation by area matching and shape preservation
    s_vals          = range(0, stop=1, length=500)                                                          # Sample points 
    # ref_crv         = BSplineCurve(discretized_set[1]; degree=2)                                               # Reference curve for area comparison
    ref_crv         = BSplineCurve(cps_list_og[1]; degree=2)
    ref_points      = [ref_crv(s) for s in s_vals]                                                          # Evaluate the reference curve at the sampled points
    reference_area  = poly_area(ref_points)                                                                 # Calculate the area of the reference polygon
    opt_cps_list    = [optimize_control_points(cps, reference_area) for cps in cps_list_og]
    cps_list        = make_symmetric_jelly(opt_cps_list)                   # Vector{Matrix{Float32}}

    new_cps_list    = reverse_cps_list(cps_list)                       # Change the cps_list from clockwise to counterclockwise order
    new_cps_list    = [hcat(start, cps[:,2:end-1], ending) for cps in new_cps_list]
    new_cps_list    = [SMatrix{2, 105, Float64, 210}(cps) for cps in new_cps_list]
return new_cps_list
end


"""
    Old functions within the geometry generation process.
"""

function add_first_point_at_end(cps_list::AbstractVector{<:SMatrix{2,N,T}}) where {N, T}
    new_cps_list = SMatrix{2, N+1, T}[]
    for cps_set in cps_list
        new_cps_set = SMatrix{2, N+1, T}(hcat(cps_set, cps_set[:, 1]))
        push!(new_cps_list, new_cps_set)
    end
    return new_cps_list
end

function remove_first_point_from_end(cps_list::AbstractVector{<:SMatrix{2,N,T}}) where {N, T}
    new_cps_list = SMatrix{2, N-1, T}[]
    for cps_set in cps_list
        new_cps_set = cps_set[:, 1:end-1]
        push!(new_cps_list, new_cps_set)
    end
    return new_cps_list
end