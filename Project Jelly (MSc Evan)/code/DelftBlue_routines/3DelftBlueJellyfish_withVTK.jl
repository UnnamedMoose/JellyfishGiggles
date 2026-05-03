using StaticArrays, WaterLily, CUDA, ParametricBodies, Interpolations, LinearAlgebra, Dierckx, BiotSavartBCs, CSV, Statistics, WriteVTK

import WaterLily: @loop,scale_u!,conv_diff!,udf!,accelerate!,BDIM!

import WaterLily: CFL
CFL(a::Flow;Δt_max=10) = 0.05

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

import BiotSavartBCs: interaction,image,symmetry
@inline function symmetry(ω,T,args...) # overwrite to add image influences
    T₁,sgn₁ = image(T,size(ω),-2)
    return interaction(ω,T,args...)+sgn₁*interaction(ω,T₁,args...)
end

fall!(flow,t;acceleration) = for i ∈ 1:ndims(flow.p)
    WaterLily.@loop flow.f[I,i] += acceleration[i] over I ∈ CartesianIndices(flow.p)
end

Tp = Float64; T = Float64

function shape_area(cps::SMatrix{2,N,T}) where {N,T}     
    sum             = zero(T)
    for i in 1:N
        x1, y1 = cps[1,i], cps[2,i]
        x2, y2 = cps[1,mod1(i+1, N)], cps[2,mod1(i+1, N)]
        sum += x1 * y2 - x2 * y1
    end
    return abs(sum) / 2
end

reverse_cps_order(cps::SMatrix{2,N,T}) where {N,T} = SMatrix{2,N,T}(cps[:, reverse(1:N)])

function refine_polyline(ctrl::SMatrix{D,M,T},N::Int) where {D,M,T}
    @assert M ≥ 2
    @assert N ≥ M "N must be ≥ number of control points"

    seglen = [norm(ctrl[:, i+1] - ctrl[:, i]) for i in 1:M-1]
    L = sum(seglen)

    nseg = round.(Int, (seglen / L) .* (N - M))
    nseg[end] += (N - M) - sum(nseg) 

    pts = Vector{SVector{D,T}}()
    push!(pts, ctrl[:,1]) 

    for i in 1:M-1
        p0 = ctrl[:, i]
        p1 = ctrl[:, i+1]

        for k in 1:nseg[i]
            α = k / (nseg[i] + 1)
            push!(pts, (1-α)*p0 + α*p1)
        end

        push!(pts, p1) 
    end

    @assert length(pts) == N
    return SMatrix{D,N,T}(hcat(pts...))
end

function optimiser(cps::SMatrix{2,N,T}, ref_area; ThreeD=false) where {N,T}
    if ThreeD == true
        area = shape_volume(cps)
    else
        area = shape_area(cps)
    end
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

function make_full_jellyfish(cps::SMatrix{2,N,T}) where {N,T}
    reverse_cps_order(cps::SMatrix{2,N,T}) where {N,T} = SMatrix{2,N,T}(cps[:, reverse(1:N)])                       
    cps_full = hcat(cps) |> reverse_cps_order
    return cps_full
end

function interpolate_cycle_duty(cps_seq; n_contr::Int, n_exp::Int, base_up::Int=10, γ::Real=1)
    Nframes = length(cps_seq)
    Npts = size(cps_seq[1], 2)
    @assert Nframes == n_contr + n_exp "length(cps_seq) must be n_contr + n_exp"

    xs = [cps_seq[t][1,j] for t in 1:Nframes, j in 1:Npts]
    ys = [cps_seq[t][2,j] for t in 1:Nframes, j in 1:Npts]

    itp_x = [interpolate(xs[:,j], BSpline(Cubic(Line(OnGrid())))) for j in 1:Npts]
    itp_y = [interpolate(ys[:,j], BSpline(Cubic(Line(OnGrid())))) for j in 1:Npts]

    up_tot = (Nframes-1) * base_up
    N_c = round(Int, γ * up_tot)
    N_e = up_tot - N_c
    γ_r = N_c / up_tot
    println("Realised γ = $γ_r, target = $γ")

    t_c = range(1, n_contr, length = N_c)[1:end]
    t_e = range(n_contr, Nframes, length = N_e)[2:end]

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
    s₀ = similar(x)    
    s₀[1] = x[1]
    for t in 2:length(x)
        s₀[t] = α * x[t] + (1 - α) * s₀[t-1]
    end
    s₁ = similar(x)    
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

function generate_jelly_motion(phase_contr, phase_exp, Ncps, T1, T2, Tg, n_cycles, n_up, γ; ThreeD=false, varyingT=false, gliding=false)
    phase_contr     = Vector{SMatrix{2,Ncps,Float64,2Ncps}}([refine_polyline(cps, Ncps) for cps in phase_contr])
    phase_exp       = Vector{SMatrix{2,Ncps,Float64,2Ncps}}([refine_polyline(cps, Ncps) for cps in phase_exp])

    if ThreeD == true
        phase_contr     = [optimiser(cps, shape_volume(phase_contr[1]); ThreeD=true) for cps in phase_contr]
        phase_exp       = [optimiser(cps, shape_volume(phase_contr[1]); ThreeD=true) for cps in phase_exp]
    else
        phase_contr     = [optimiser(cps, shape_area(phase_contr[1])) for cps in phase_contr]
        phase_exp       = [optimiser(cps, shape_area(phase_contr[1])) for cps in phase_exp]
    end

    cps_seq         = Vector{SMatrix{2,Ncps,Float64,2Ncps}}(vcat(phase_contr, phase_exp))

    period_frames_T1    = interpolate_cycle_duty(cps_seq; n_contr=4, n_exp=6, base_up=n_up, γ=γ)
    n_up_gliding        = round(Int, length(period_frames_T1) * Tg / T1)
    gliding_frames      = fill(period_frames_T1[1], n_up_gliding)

    if varyingT == false && gliding == false
        period_frames       = period_frames_T1
        frames              = vcat( (period_frames for _ in 1:n_cycles)... )
    elseif varyingT == true && gliding == false
        period_frames_T2    = interpolate_cycle_uniform(cps_seq; base_up=Int((T2/T1)*n_up))
        period_frames       = vcat(period_frames_T1, period_frames_T2)
        frames              = repeat(period_frames, n_cycles)
    elseif varyingT == false && gliding == true
        period_frames       = vcat(period_frames_T1, gliding_frames)
        frames              = repeat(period_frames, n_cycles)
    elseif varyingT == true && gliding == true
        period_frames_T2    = interpolate_cycle_uniform(cps_seq; base_up=Int((T2/T1)*n_up))
        period_frames       = vcat(period_frames_T1, gliding_frames, period_frames_T2, gliding_frames)
        frames              = repeat(period_frames, n_cycles)
    end

    frames          = [make_full_jellyfish(cps) for cps in frames]
    Nframes         = length(frames)
    Npts            = size(frames[1],2)
    sx              = [[frames[t][1,i] for t=1:Nframes] for i=1:Npts]
    sy              = [[frames[t][2,i] for t=1:Nframes] for i=1:Npts]
    len             = length(sx)
    sx_smooth       = [exp_smooth(sx[i], 0.25) for i in 1:len]
    sy_smooth       = [exp_smooth(sy[i], 0.25) for i in 1:len]


    if varyingT == false && gliding == false
        time_set    = range(0,T1 * n_cycles, length=length(sx_smooth[1]))
    elseif varyingT == true && gliding == false
        time_set    = range(0,(T1+T2) * n_cycles,length=length(sx_smooth[1]))   
    elseif varyingT == false && gliding == true
        time_set    = range(0, (T1+Tg) * n_cycles, length=length(sx_smooth[1]))
    elseif varyingT == true && gliding == true
        time_set    = range(0,(T1+T2+Tg) * n_cycles,length=length(sx_smooth[1]))   
    end

    pathing         = control_point_functions(sx_smooth, sy_smooth, time_set)
    return pathing
end

knots_vector(p::Int, Ncp::Int) = vcat(zeros(p+1), (Ncp-p-1 > 0 ? collect(range(0.0, 1.0, length=Ncp-p+1))[2:end-1] : Float64[]), ones(p+1))

scratch_dir = "/scratch/$(get(ENV,"USER","unknown"))/jellyfish_runs/base_case"
isdir(scratch_dir) || mkpath(scratch_dir)

jobid = get(ENV, "SLURM_JOB_ID", "interactive")

vtk_velocity(a::AbstractSimulation) = a.flow.u |> Array
vtk_pressure(a::AbstractSimulation) = a.flow.p |> Array
vtk_vorticity(a::AbstractSimulation) = (@inside a.flow.σ[I] = WaterLily.curl(3,I,a.flow.u)*a.L/a.U; a.flow.σ |> Array)
vtk_body(a::AbstractSimulation) = (measure_sdf!(a.flow.σ, a.body, WaterLily.time(a.flow)); a.flow.σ |> Array)
function vtk_laplacian(a::AbstractSimulation)
    L = copy(a.flow.μ₁); N = length(size(a.flow.μ₁))
    WaterLily.@loop L[I,:,:] .= WaterLily.S(I,a.flow.u) over I in WaterLily.inside_u(a.flow.u)
    return permutedims(L,[N,1:N-1...]) |> Array 
end
custom_write_attributes = Dict("u" => vtk_velocity,
                            "p" => vtk_pressure,
                            "ω" => vtk_vorticity,
                            "∇²u" => vtk_laplacian,
                            "d" => vtk_body)

wr = vtkWriter("flowfield_" * jobid;
               dir = scratch_dir,
               attrib = custom_write_attributes)

function jelly_simulation_3D(duration, D, period, Re, U, ϵ, pathing, Uff, Ncps)
    rev_map(x,t)    = SA[(x[1]), hypot(x[2], x[3])] 
    cps         =   cps_at_time(pathing, Ncps, 0;) .* D .+ SA{T}[0.5D; 0]
    weights     =   ones(T, size(cps, 2)); knots = Float64.(knots_vector(deg, size(cps, 2))); curve = NurbsCurve(cps, knots, weights )
    body        =   ParametricBody(curve; map=rev_map, ndims=3)
    sim         =   BiotSimulation((9D, D, D), (Uff,Uff,Uff), D; U, ν=(U*D) / Re, body, T, mem=CuArray, ϵ, nonbiotfaces=(-2,-3))

    t₀ = round(sim_time(sim)); step = 0.1; t0 = 0; a0 = 0; v0 = 0.f0; Re_new = 0; p0 = 0; f0 = 0; τ = 0.03 * period * sim.L/sim.U;    
    results_file = joinpath(@__DIR__, "results_3D_$period.csv")

    open(results_file, "w") do io
        println(io, "fpres,fvisc,fam,fdrag,time,acceleration,velocity,position,volume")
        for tᵢ in range(t₀, t₀ + (duration * period); step)        
            t = sum(sim.flow.Δt[1:end-1])
            while t < tᵢ * sim.L / sim.U
                cps             = cps_at_time(pathing, Ncps, t* sim.U/sim.L) .* D .+ SA{T}[0.5D; 0]
                sim.sim.body    = ParametricBodies.update!(sim.sim.body, cps, sim.flow.Δt[end])

                measure!(sim)
                biot_mom_step!(sim.flow,sim.pois,sim.ω,sim.x₀,sim.tar,sim.ftar;
                               fmm=sim.fmm,udf=fall!,acceleration=SA[-a0,0.f0,0.f0],U=SA[-v0,0.f0,0.f0])
                
                if t < 0.2* period * sim.L/sim.U
                    force_can = 0
                    Re_new = 302
                else
                    force_can           =   -4*WaterLily.total_force(sim)[1]
                    Re_new          =   -(v0 * D)/ ((U*D) / Re)
                end
                
                Δt              =   sim.flow.Δt[end]
                vol             =   shape_volume(cps)                       # non-dimensionally: V_nd = V/(D^3)
                Fpres           =   -4*WaterLily.pressure_force(sim)[1]
                Fvisc           =   -4*WaterLily.viscous_force(sim)[1]
                Fam             =   0.5*vol*a0
                
                Fdrag           =   0.5*(24/Re_new^0.7)*π*(32)^2*v0^2
                
                α               =   1 - exp(-Δt / (τ))
                force           =   (1-α)*f0 + α *force_can
                accel           =   (Fpres + Fdrag) / (vol*(1+0.5))                           # non-dimensionally: a_nd = f/(U^2D^2) / V/(D^3) = (fD^3)/(VU^2D^2) = f/V * (D/U^2), D/U^2 and U^2/D = acceleration scaling so correct.

                p0              +=  Δt * (v0 + Δt * accel / 2.)
                v0              +=  Δt * accel                              # non-dimensionally: v_nd = (ΔtU/D) * (aD/U^2) = (Δta)/U, and again U = velocity scaling so correct.
                a0              =   accel
                f0              =   force
                tnum            =   t * sim.U/sim.L

                if !isfinite(force)
                    println("Diverging Solution")
                end

                println(io, "$Fpres,$Fvisc,$Fam,$Fdrag,$tnum,$a0,$v0,$p0,$vol")

                t0 = t; t += Δt
            end
            save!(wr, sim)
            @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L/sim.U
            @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I]) < 0.001, 0.0, sim.flow.σ[I])
            vort = maximum(sim.flow.σ)
            println("tU/L=", round(tᵢ, digits = 4), ", Δt=", round(sim.flow.Δt[end], digits = 3), ", om=", round(maximum(vort)))
        end 
    end
    close(wr)
end
cps_0 = SA{Float64}[0.000000000   0.024608501   0.107382550   0.250559284   0.413870246   0.601789709   0.868008949   1.078299776   1.199105145   1.232662192   1.210290828   1.153310962   1.170022371   1.102908277   0.800894855   0.599552573   0.445190157   0.362416107   0.333333333   0.326621924
    0.000000000   0.202191891   0.399758691   0.534268406   0.603564337   0.625613956   0.600296609   0.548138652   0.516406505   0.460151321   0.206269009   0.206374582   0.424337028   0.453701330   0.449885630   0.414383028   0.315853505   0.201432773   0.109363296   0.000000000]
cps_1 = SA{Float64}[0.000000000   0.029082774   0.114093960   0.255033557   0.418344519   0.601789709   0.856823266   1.071588367   1.183445190   1.223713647   1.313199105   1.266219239   1.156599553   1.102908277   0.800894855   0.599552573   0.451901566   0.369127517   0.340044743   0.331096197
    0.000000000   0.202181836   0.395249227   0.523022397   0.590071136   0.609883619   0.568861071   0.503209914   0.480486640   0.421969183   0.212779328   0.188165799   0.401895282   0.415499083   0.431908101   0.394158309   0.309096850   0.199170500   0.107101023   0.000000000]
cps_2 = SA{Float64}[0.000000000   0.031319911   0.123042506   0.259507830   0.420581655   0.601789709   0.843400447   1.060402685   1.178970917   1.217002237   1.395973154   1.369127517   1.170022371   1.102908277   0.800894855   0.599552573   0.469798658   0.389261745   0.355704698   0.342281879
    0.000000000   0.202176809   0.386240354   0.507282005   0.574335772   0.585164518   0.528441797   0.435819320   0.406339391   0.343332579   0.176638263   0.138496343   0.314224669   0.332353015   0.375728326   0.358203253   0.282090340   0.187889299   0.100324259   0.000000000]
cps_3 = SA{Float64}[0.000000000   0.035794183   0.129753915   0.261744966   0.422818792   0.601789709   0.836689038   1.060402685   1.176733781   1.223713647   1.355704698   1.319910515   1.167785235   1.100671141   0.800894855   0.599552573   0.480984340   0.407158837   0.371364653   0.353467562
    0.000000000   0.202166755   0.379483699   0.496041023   0.558600407   0.564939798   0.508232159   0.413347410   0.386119699   0.327587160   0.152009652   0.118382224   0.289510595   0.305391750   0.337526079   0.324495387   0.257346103   0.174365935   0.093547495   0.000000000]
cps_4 = SA{Float64}[0.000000000   0.035794183   0.125279642   0.259507830   0.420581655   0.601789709   0.843400447   1.069351230   1.176733781   1.223713647   1.230425056   1.180156600   1.176733781   1.102908277   0.800894855   0.599552573   0.476510067   0.400447427   0.366890380   0.351230425
    0.000000000   0.204413946   0.381740945   0.507282005   0.567594199   0.576175753   0.530688988   0.471754267   0.449041048   0.386014127   0.183751854   0.181595154   0.361400598   0.377296835   0.373481135   0.342472916   0.268592112   0.181122590   0.095804741   0.000000000]
cps_5 = SA{Float64}[0.000000000   0.035794183   0.114093960   0.255033557   0.416107383   0.601789709   0.861297539   1.073825503   1.192393736   1.217002237   1.167785235   1.123042506   1.161073826   1.102908277   0.800894855   0.599552573   0.456375839   0.378076063   0.346756152   0.340044743
    0.000000000   0.204413946   0.388507654   0.523022397   0.596817736   0.612130810   0.582334163   0.534665561   0.498444059   0.444456175   0.215353292   0.231184174   0.424357137   0.444712566   0.436402483   0.391911118   0.302345223   0.194656009   0.104838750   0.000000000]
cps_6 = SA{Float64}[0.000000000   0.031319911   0.111856823   0.252796421   0.409395973   0.601789709   0.868008949   1.078299776   1.196868009   1.221476510   1.174496644   1.117516779   1.163310962   1.102908277   0.798657718   0.599552573   0.447427293   0.366890380   0.335570470   0.335570470
    0.000000000   0.204424000   0.390759872   0.527526833   0.610315964   0.625613956   0.604790991   0.552633034   0.511917151   0.460176457   0.210843828   0.226679738   0.426599301   0.449206948   0.456632230   0.416630219   0.313601287   0.199175527   0.107111078   0.000000000]
cps_7 = SA{Float64}[0.000000000   0.031319911   0.109619687   0.250559284   0.407158837   0.601789709   0.865771812   1.080536913   1.196868009   1.221476510   1.185682327   1.125413870   1.163310962   1.102908277   0.800894855   0.599552573   0.449664430   0.369127517   0.340044743   0.333333333
    0.000000000   0.204424000   0.390764900   0.532026242   0.612568182   0.630108338   0.600301636   0.552628007   0.511917151   0.460176457   0.208571500   0.215403564   0.426599301   0.449206948   0.452132821   0.409888646   0.311349069   0.199170500   0.107101023   0.000000000]
cps_8 = SA{Float64}[0.000000000   0.029082774   0.107382550   0.248322148   0.411633110   0.601789709   0.863534676   1.080536913   1.196868009   1.221476510   1.206868009   1.148836689   1.165548098   1.102908277   0.800894855   0.599552573   0.447427293   0.366890380   0.337807606   0.331096197
    0.000000000   0.204429027   0.390769927   0.534268406   0.605816555   0.632355529   0.595812282   0.554875198   0.509669960   0.460176457   0.206299173   0.210879019   0.428841465   0.455948521   0.452132821   0.412135837   0.313601287   0.199175527   0.107106050   0.000000000]
cps_9 = SA{Float64}[0.000000000   0.026845638   0.102908277   0.250559284   0.411633110   0.601789709   0.870246085   1.078299776   1.201342282   1.228187919   1.212393736   1.148836689   1.165548098   1.102908277   0.800894855   0.601789709   0.445190157   0.362416107   0.333333333   0.326621924
    0.000000000   0.204434055   0.390779981   0.525274615   0.608063746   0.623366765   0.607033155   0.548138652   0.516401478   0.460161375   0.207320464   0.211890255   0.428841465   0.451454139   0.445391248   0.416625192   0.315853505   0.201432773   0.109363296   0.000000000] 

Ncps            = 35              
n_cycles        = 100                
n_up            = 20                
γ               = 4/10               
duration        = 20             

D               = 2^6               
ϵ               = 2                 
deg             = 1                 
Uff             = 0                 
U               = 1                 

Dmax            = 1.25              
Uavg            = 2.42              
Re              = 302               
T1              = 1 * Uavg / Dmax
T2              = 2 * Uavg / Dmax
Tg              = 0.5 * Uavg / Dmax
TgT1            = Tg / T1 

phase_contr     = [cps_0, cps_1, cps_2, cps_3] ./ Dmax               
phase_exp       = [cps_4, cps_5, cps_6, cps_7, cps_8, cps_9] ./ Dmax        
pathing  = generate_jelly_motion(phase_contr, phase_exp, Ncps, T1, T2, Tg, n_cycles, n_up, γ; ThreeD=true, varyingT=false, gliding=false)

jelly_simulation_3D(duration, D, T1, Re, U, ϵ, pathing, Uff, Ncps)