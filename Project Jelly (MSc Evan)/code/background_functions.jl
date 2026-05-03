import WaterLily: @loop,scale_u!,conv_diff!,udf!,accelerate!,BDIM!,CFL,sim_step!
import BiotSavartBCs: biot_mom_step!,biot_project!

"""
Adjust the WaterLily numerical time step. Generally, it is restricted to 0.10 or lower for finer grid sizes for this specific problem.
"""
CFL(a::Flow;Δt_max=10) = 0.05

"""
Not really sure if this is still used as I use the Biot-Savart updater. But that one may also be dependent on the sim_step updater.
"""
function sim_step!(sim::AbstractSimulation,t_end;remeasure=true,λ=quick,max_steps=typemax(Int),verbose=false,
        udf=nothing,kwargs...)
    steps₀ = length(sim.flow.Δt)
    while sim_time(sim) < t_end && length(sim.flow.Δt) - steps₀ < max_steps
        sim_step!(sim; remeasure, λ, udf, kwargs...)
        verbose && sim_info(sim)
    end
end

"""
Adjust the Biot-Savart updater to the version with a moving grid, so that the velocity and acceleration vectors can be applied to the updater.
"""
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

"""
Apply the acceleration to the flow field.
"""
fall!(flow,t;acceleration) = for i ∈ 1:ndims(flow.p)
    WaterLily.@loop flow.f[I,i] += acceleration[i] over I ∈ CartesianIndices(flow.p)
end

"""
Some old function for computing phase averages.
"""
function phase_average(signal::Vector{Float64}, period::Int)
    n_periods = div(length(signal), period)
    reshaped = reshape(signal[1:n_periods * period], period, n_periods)
    mean(reshaped, dims=2)[:]
end

"""
Power computation functions. All input should be adjusted case-specifically to acquire a representative power.
    - `pathing` = the matrix with interpolation functions of the control points
    - `Ncps` = number of control points
"""
function compute_power(pathing, Ncps; period = period, Δt = 0.10, U = 1, D = 64, γ = 4/10, time_vector = time1, force_vector = force1)
    phase                   = mod.(time_vector, period) ./ period  
    mask                    = phase .< γ
    period_index            = floor.(time_vector ./ period)
    n_periods               = Int(maximum(period_index))
    dt                      = time_vector[2] - time_vector[1]           # = Δt/D units

    mean_power_per_period   = Vector{Float64}(undef, n_periods)
    inst_power_per_period   = Vector{Vector{Float64}}(undef, n_periods)

    for p in 0:n_periods-1
        inds = findall((period_index .== p) .& mask)

        P_inst = Float64[]

        for i in inds[1:end-1]
            cps0 = SMatrix{2,Ncps}(cps_at_time(pathing, Ncps, time_vector[i]) .* D)
            cps1 = SMatrix{2,Ncps}(cps_at_time(pathing, Ncps, time_vector[i+1]) .* D)

            cp_vel = mean((cps1 - cps0) / dt) 
            push!(P_inst, force_vector[i] * cp_vel)
        end

        inst_power_per_period[p+1] = P_inst
        mean_power_per_period[p+1] = mean(P_inst)
    end

    return mean_power_per_period, inst_power_per_period
end

"""
Structure to define simulation data according to the CSV output of the simulations.
"""
struct SimulationData
    time   :: Vector{Float64}
    force  :: Vector{Float64}
    vel    :: Vector{Float64}
    acc    :: Vector{Float64}
    pos    :: Vector{Float64}
    vol    :: Vector{Float64}
end

"""
Function to load a simulation CSV results file and acquire the data from it.
"""
function load_simulation(file)
    df = CSV.read(file, DataFrame)
    return SimulationData(
        df.time,
        df.forces,
        df.velocity,
        df.acceleration,
        df.position,
        df.volume
    )
end

"""
Function to find the stationary part of the signal arrays. The point at which it is statistically stationary based on windowed mean and standard deviation convergence. Adjust the tolerances to change the strictness of the searcher. It searches for 3 consecutive cycles that are within the tolerances regarding the mean and standard deviation of the signal. Statistically stationary basically means here that the mean and standard deviation of a window (cycle) stop changing significantly over time (3 cycles).

    - `signal::AbstractVector` = the signal input array
"""
function find_stationary_index(
    signal::AbstractVector;
    window_size::Int,
    tol_mean::Float64=1e-3,
    tol_std::Float64=1e-3,
    n_consecutive::Int=3
)
    n = length(signal)
    nwin = fld(n, window_size)

    means = zeros(nwin)
    stds = zeros(nwin)

    # Compute window statistics
    for i in 1:nwin
        w = signal[(i-1)*window_size+1:i*window_size]
        means[i] = mean(w)
        stds[i] = std(w)
    end

    count = 0
    for i in 2:nwin
        dmean = abs(means[i] - means[i-1]) / max(abs(means[i-1]), eps())
        dstd = abs(stds[i] - stds[i-1]) / max(stds[i-1], eps())

        if dmean < tol_mean && dstd < tol_std
            count += 1
            if count ≥ n_consecutive
                # Return index in original signal
                return (i - n_consecutive) * window_size + 1
            end
        else
            count = 0
        end
    end

    return nothing  # no stationary region found
end