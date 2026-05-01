"""
Function to reverse the order of the jellyfish, can also comment on the commands to mirror the jellyfish and acquire the control points for a full jellyfish shape.
"""
function make_full_jellyfish(cps::SMatrix{2,N,T}) where {N,T}
    reverse_cps_order(cps::SMatrix{2,N,T}) where {N,T} = SMatrix{2,N,T}(cps[:, reverse(1:N)])                       
    # cps_sym = SMatrix{2,N-1,Float64,2(N-1)}(cps[:,1:end-1] .* [1; -1]) |> reverse_cps_order
    # cps_full = hcat(cps, cps_sym) |> reverse_cps_order
    # cps_full = hcat(cps, cps[:,1]) |> reverse_cps_order
    cps_full = hcat(cps) |> reverse_cps_order
    return cps_full
end

"""
Function to generate the knots vector, uniformly clamped.
"""
knots_vector(p::Int, Ncp::Int) = vcat(zeros(p+1), (Ncp-p-1 > 0 ? collect(range(0.0, 1.0, length=Ncp-p+1))[2:end-1] : Float64[]), ones(p+1))

"""
Function for area computation through the shoelace formula.
"""
function shape_area(cps::SMatrix{2,N,T}) where {N,T}     
    sum             = zero(T)
    for i in 1:N
        x1, y1 = cps[1,i], cps[2,i]
        x2, y2 = cps[1,mod1(i+1, N)], cps[2,mod1(i+1, N)]
        sum += x1 * y2 - x2 * y1
    end
    return abs(sum) / 2
end

"""
Function for volume computation with the disk method.
"""
function shape_volume(cps::SMatrix{2,N,T}) where {N,T}
    V = zero(T)

    for i in 1:(N-1)
        x1, y1 = cps[1,i], cps[2,i]
        x2, y2 = cps[1,i+1], cps[2,i+1]

        dx = (x2 - x1)
        V += dx * (y1^2 + y1*y2 + y2^2)
    end

    return abs(π * V / 3)
end