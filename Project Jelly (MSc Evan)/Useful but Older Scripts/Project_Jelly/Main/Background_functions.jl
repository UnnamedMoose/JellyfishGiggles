using StaticArrays, Plots, WaterLily, ParametricBodies, Interpolations, LinearAlgebra, Dierckx, GLMakie, Images, ImageMagick, ImageIO, BiotSavartBCs, DelimitedFiles, DataFrames, CSV, Statistics

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


function make_full_jellyfish(cps::SMatrix{2,N,T}) where {N,T}
    reverse_cps_order(cps::SMatrix{2,N,T}) where {N,T} = SMatrix{2,N,T}(cps[:, reverse(1:N)])                       
    cps_sym = SMatrix{2,N-1,Float64,2(N-1)}(cps[:,1:end-1] .* [1; -1]) |> reverse_cps_order
    cps_full = hcat(cps, cps_sym) |> reverse_cps_order
    return cps_full
end

function shape_area(cps::SMatrix{2,N,T}) where {N,T}
    s_vals          = range(0, 1; length=100)            
    curve           = BSplineCurve(cps; degree=1)
    points          = [curve(s) for s in s_vals]
    n               = length(points)
    sum             = zero(T)
    for i in 1:n
        x1, y1 = points[i]
        x2, y2 = points[mod1(i+1, n)]
        sum += x1 * y2 - x2 * y1
    end
    return abs(sum) / 2
end

function resample_by_arclength(curve, N::Int; nsample::Int = 500)

    # 1. Sample the curve uniformly in parameter space
    points                 = map(curve, range(0.0, 1.0; length = nsample))

    # 2. Compute cumulative arc length
    arc_length          = cumsum(vcat(0.0, norm.(diff(points))))
    total_length        = arc_length[end]
    # 3. Desired equally spaced arc-length positions (uses equal spacing)
    target_lengths      = range(0.0, total_length; length = N)

    # 4. Invert arc-length → parameter mapping
    resampled              = LinearInterpolation(arc_length, range(0.0, 1.0; length = nsample), extrapolation_bc = Flat()).(target_lengths)

    # 5. Evaluate curve at resampled parameters
    resampled_pts       = map(curve, resampled)

    return SMatrix{2,N,Float64}(hcat(resampled_pts...))
end

function optimiser(cps::SMatrix{2,N,T}, ref_area) where {N,T}
    area = shape_area(cps)
    s = √(ref_area / area)      
    cx = sum(cps[1,i] for i in 1:N) / N
    cy = sum(cps[2,i] for i in 1:N) / N
    cpsn = MMatrix{2,N,T}(cps)

    for i in 1:N
        x = cps[1,i]
        y = cps[2,i]
        cpsn[1,i] = cx + s*(x - cx)
        cpsn[2,i] = cy + s*(y - cy)
    end

    return SMatrix{2,N,T}(cpsn)
end

function interpolate_cycle_duty(cps_seq; n_contr::Int, n_exp::Int, base_up::Int=10, γ::Real=1.0)
    Nframes = length(cps_seq)
    @assert Nframes == n_contr + n_exp "length(cps_seq) must be n_contr + n_exp"

    Npts = size(cps_seq[1], 2)

    # Extract time series: (time × point_index)
    xs = [cps_seq[t][1,j] for t in 1:Nframes, j in 1:Npts]
    ys = [cps_seq[t][2,j] for t in 1:Nframes, j in 1:Npts]

    # Build cubic B-spline interpolators over t = 1:Nframes
    itp_x = [interpolate(xs[:,j], BSpline(Cubic(Line(OnGrid())))) for j in 1:Npts]
    itp_y = [interpolate(ys[:,j], BSpline(Cubic(Line(OnGrid())))) for j in 1:Npts]

    # Choose upsample factors to realise duty cycle γ
    up_c = base_up
    up_e_float = γ * (n_contr-1) / (n_exp-1) * up_c
    up_e = max(1, round(Int, up_e_float))

    # Build time grids for contraction and expansion (one cycle)
    # Contraction: frames 1 .. n_contr
    n_c_samples = (n_contr-1)*up_c + 1

    # Maybe I want this in time: initial contraction phase is from 0 -> (4*period)/10
    # And expansion phase is from (4*period)/10 -> (T)

    t_c = range(1, n_contr, length = n_c_samples)[1:end-1]
    n_e_samples = (n_exp-1)*up_e + 1
    t_e = range(n_contr, Nframes, length = n_e_samples)

    t_cycle = [t_c; t_e]   

    # Evaluate interpolants on this non-uniform time grid
    out = Vector{SMatrix{2,Npts,Float64,2Npts}}(undef, length(t_cycle))

    for (k, τ) in enumerate(t_cycle)
        M = @MMatrix zeros(Float64, 2, Npts)
        @inbounds for j in 1:Npts
            M[1,j] = itp_x[j](τ)
            M[2,j] = itp_y[j](τ)
        end
        out[k] = SMatrix{2,Npts,Float64,2Npts}(M)
    end

    return out
end

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

cps_at_time(interp_funcs, Npoints, frame) = SMatrix{2,Npoints,Float64}(hcat([f(frame) for f in interp_funcs]...) )

"""
Control point sets as digitised from (Sahin 2009), dividing the coordinates by the characteristic length used in the paper (Dmax).
Digitised using automeris.io.
Static Arrays are required for ParametricBodies, first row is x-coordinate, second row is y-coordinate.
Best to continuously assign Static Arrays including size and type to avoid erros. (SMatrix{rows,columns,type}())
"""

function generate_jelly_motion(phase_contr, phase_exp, Ncps, n_cycles, n_up, γ)
    """
    Step 1: Discretise into way more points using a BSpline Curve of an accurate degree. 
    For the simulations, degree = 1 is required, so seems the most logical choice for discretisation as well.
    If I reconstruct a curve using these points, the first difference w.r.t. the original occurs. This is due to the knots representative of BSpline Curves.
    I might need a method to resample accurately, also considering the influence of the weights and knots.
    """

    phase_contr     = Vector{SMatrix{2,Ncps,Float64,2Ncps}}([resample_by_arclength(BSplineCurve(cps; degree=1), Ncps) for cps in phase_contr])
    phase_exp       = Vector{SMatrix{2,Ncps,Float64,2Ncps}}([resample_by_arclength(BSplineCurve(cps; degree=1), Ncps) for cps in phase_exp])

    """
    Step 2: Acquire area conservation (aka mass conservation)
    Simple algorithm using the initial shape area as reference and relocating the control points to minimise the fraction between areas.
    """

    phase_contr     = [optimiser(cps, shape_area(phase_contr[1])) for cps in phase_contr]
    phase_exp       = [optimiser(cps, shape_area(phase_contr[1])) for cps in phase_exp]
    cps_seq         = Vector{SMatrix{2,Ncps,Float64,2Ncps}}(vcat(phase_contr, phase_exp))

    """
    Step 3: Upsampling of the control point sets. The number of frames is expanded.
    A duty cycle parameter is implemented here, which is used to upsample contraction and expansion frames apart.
    Thus, the duty cycle is directly implemented into the resulting set of frames.
    The periodic frames are then repeated n_cycles number of times.
    """

    period_frames   = interpolate_cycle_duty(cps_seq; n_contr=length(phase_contr), n_exp=length(phase_exp), base_up=n_up, γ=γ)
    frames          = vcat( (period_frames for _ in 1:n_cycles)... )

    """
    Make the full symmetrical jellyfish and reverse the control points to acquire a flow around the body instead of inside (WL definitions).
    Transform the control point sets into individual control point pathing arrays. 
    Apply an exponential smoothing algorithm to the pathings. This is required as otherwise the connection between 2 cycles and between contraction-expansion is too sharp for WaterLily.
    """

    frames          = [make_full_jellyfish(cps) for cps in frames]
    Nframes         = length(frames)
    Npts            = size(frames[1],2)
    sx              = [[frames[t][1,i] for t=1:Nframes] for i=1:Npts]
    sy              = [[frames[t][2,i] for t=1:Nframes] for i=1:Npts]
    len             = length(sx)
    sx_smooth       = [exp_smooth(sx[i], 0.25) for i in 1:len]
    sy_smooth       = [exp_smooth(sy[i], 0.25) for i in 1:len]

    """
    Translate the control point pathings into 'time' dependent functions for each control point individually.
    Time is a bit of a weird term, as the actual input here is in terms of frame #.
    First and last period can be a bit 'off', so better to not use these and use period 2 to end-1.
    """

    frame_points    = range(1, length(sx_smooth[1]), step=1)
    pathing         = control_point_functions(sx_smooth, sy_smooth, frame_points)
    t_fr_ratio      = length(sx_smooth[1]) / n_cycles 

    return pathing, t_fr_ratio
end

knots_vector(p::Int, Ncp::Int) = vcat(zeros(p+1), (Ncp-p-1 > 0 ? collect(range(0.0, 1.0, length=Ncp-p+1))[2:end-1] : Float64[]), ones(p+1))

sample_signal(f, tspan) = f.(tspan)

fall!(flow,t;acceleration) = for i ∈ 1:ndims(flow.p)
    WaterLily.@loop flow.f[I,i] += acceleration[i] over I ∈ CartesianIndices(flow.p)
end

function run_jelly_simulation(period, period_fr, D, Re, U, ϵ, pathing, Domain, Uff, Ncps)
    xloc = Domain / 6; yloc = Domain/2; Domain_y = Domain 
    cps         =   cps_at_time(pathing, 2*Ncps-1, 0;) .* D .+ SA{T}[xloc, yloc] 
    weights     =   ones(T, size(cps, 2)); knots = Float64.(knots_vector(deg, size(cps, 2))); curve = NurbsCurve(cps, knots, weights )
    body        =   DynamicNurbsBody(curve; thk=0, boundary=true)
    sim         =   BiotSimulation((Domain, Domain_y), (Uff,Uff), D; U, ν, body, T, mem=Array, ϵ)
    duration    = 25; t₀ = round(sim_time(sim)); step = 0.1; t0 = 0; a0 = 0; v0 = 0; p0 = 0

    open("Data/Simulation_Data/results.csv", "w") do io
        println(io, "forces,force_addedmass,force_inertia,force_drag,time,displacement,velocity,acceleration,enstrophy")
        for tᵢ in range(t₀, t₀ + duration; step)        
            t = sum(sim.flow.Δt[1:end-1])
            while t < tᵢ * sim.L / sim.U
                cps             = cps_at_time(pathing, 2*Ncps-1, t*(period_fr/(period))) .* D .+ SA{T}[xloc, yloc] 
                d = maximum(cps[2,:]) - yloc; h = maximum(cps[1,:]) - xloc; α = (2*h / d)^1.4
                sim.sim.body    = ParametricBodies.update!(sim.sim.body, cps, sim.flow.Δt[end])
                # sim_step!(sim, t/sim.L; remeasure = true)
                measure!(sim)
                biot_mom_step!(sim.flow,sim.pois,sim.ω,sim.x₀,sim.tar,sim.ftar;
                               fmm=sim.fmm,udf=fall!,acceleration=SA[-a0,0.f0],U=SA[-v0,0.0]) # change of frame

                # Hollow hemisphere -> Substract the 'inside' hemisphere from the outside one -> Is what I do now!
                # Wouldnt it be a more viable option to see the body + inside fluid as a 'rigid body' that changes in size/shape. The fluid moves with the jelly so...
                # In the paper of Gabe, he assumes m to be constant but V is subject to isotropic change.
                
                ## Force Balance as in (Daniel 1982), jet model:
                # Thrust = Drag + acc reaction + force to overcome body inertia 
                # Thrust = uₑ(dm/dt) with uₑ the velocity of ejected fluid and m the instantaneous mass of animal + fluid in cavity. 
                # Drag = Cd(0.5ρSu²) with Cd = 24/Reⁿ with n=0.7 for Re up to 500, S is projected area and u is inst. velocity of the jellyfish.
                # Acc reaction = αρVa           -> αA(a-a₀)
                # Inertia Force = ρVa           -> Aa
                ## In WaterLily this would be ???
                # Total Hydrodynamic Force = Aa + αA(a - a₀) + ...
                # So how to fill in the dots. Some sort of drag and thrust?

                force           =   -WaterLily.pressure_force(sim)[1]
                Δt              =   sim.flow.Δt[end]
                force_dr        =   24 / (Re^(0.7)) * 0.5 * shape_area(cps) * v0
                accel           =   (force + α * shape_area(cps) * a0) / (shape_area(cps) * (1 + α))
                force_in        =   shape_area(cps) * accel
                force_am        =   α * shape_area(cps) * (accel - a0)
                
                p0              +=  Δt * (v0 + Δt * accel / 2.)
                v0              +=  Δt * accel
                a0              =   accel
                tnum            =   t * sim.U/sim.L

                if !isfinite(force)
                    println("Diverging Solution")
                end

                @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L/sim.U
                @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I]) < 0.001, 0.0, sim.flow.σ[I])
                ω               =   Array(sim.flow.σ)
                enstr           =   sum(ω.^2)

                println(io, "$force,$force_am,$force_in,$force_dr,$tnum,$p0,$v0,$a0,$enstr")

                t0 = t; t += sim.flow.Δt[end]
            end

            gen_p_plots(sim, tᵢ, Domain)
            gen_u_plots(sim, tᵢ, Domain)
            gen_ω_gif(sim, tᵢ, Domain)
            
            println("tU/L=", round(tᵢ, digits = 4), ", Δt=", round(sim.flow.Δt[end], digits = 3))
        end 
    end
end

function make_periodic_signal(t, y)
    τ = t .- t[1]
    T = τ[end]

    spline = Spline1D(τ, y; k=3, bc="extrapolate")

    periodic(tval) = spline(mod(tval, T))

    return periodic, T
end

function make_window_signal(t, y)
    τ = t .- t[1]            
    T = τ[end]             

    itp = LinearInterpolation(τ, y)

    window(t0) = tval -> itp(tval - t0)

    return window, T
end

make_periodic_from(data) = begin
    t, y = data[1,:], data[2,:]
    make_periodic_signal(t, y)
end

make_window_from(data) = begin
    t, y = data[1,:], data[2,:]
    make_window_signal(t, y)
end

