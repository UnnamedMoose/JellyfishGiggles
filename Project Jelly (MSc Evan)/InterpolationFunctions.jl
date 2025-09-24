
@inline function interpolate_cps(new_cps_list, t::Tp, Δt::Tp, sim, v::Tp, s::Tp; nphases::Int=10)    
    period = Tp(3) * sim.L / sim.U
    τ_total = t / period
    # @show τ_total
    k = floor(Int, τ_total * nphases)
    # @show k
    τ_local = τ_total * nphases - k
    cps_a = new_cps_list[mod(k, nphases) + 1]
    cps_b = new_cps_list[mod(k+1, nphases) + 1]
    # interp_metric = ((Tp(1) - τ_local) .* cps_a .+ τ_local .* cps_b)
    
    # force = periodic_force
    # @show force
    # @show WaterLily.pressure_force(sim)[1]
    # volume = (get_area(cps_a) * Tp(2) * π * maximum(cps_a[2,:]))
    # @show volume

    # a = force / volume
    # v::Tp = v + a[end]*Δt*(sim.U / sim.L)
    # s::Tp = s + v *Δt*(sim.U/sim.L) 
    # @show t
    # mov = 0.1 * t
    offset = SVector{2, Tp}(4sim.L, 3sim.L)
    # interpolated = ((Tp(1) - τ_local) .* cps_a .+ τ_local .* cps_b) .* sim.L .+ offset
    area = get_area(interpolated ./ sim.L)
    # push!(areas, area)
    # push!(τ_locals, τ_local)
    interpolated = cps_b .* sim.L .+ offset
    # @show interpolated
    return interpolated
end

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

@inline function interpolate_cps_hermite_new(new_cps_list, t::Tp, period; nphases::Int=10) where Tp
    τ_total = t / period

    k = floor(Int, τ_total * nphases)
    τ_local = τ_total * nphases - k

    idx0 = mod(k, nphases) + 1
    idx1 = mod(k + 1, nphases) + 1
    idx_prev = mod(k - 1, nphases) + 1
    idx_next = mod(k + 2, nphases) + 1
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

    interpolated = (h00 .* p0 .+ h10 .* m0 .+ h01 .* p1 .+ h11 .* m1) 

    return interpolated
end