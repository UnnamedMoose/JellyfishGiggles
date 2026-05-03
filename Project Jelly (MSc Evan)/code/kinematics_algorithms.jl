"""
Function to quickly evaluate the control point matrix from the pathing function `pathing` on a specific time step. It then turns this into a static matrix that can be used to create a Parametric Body in WaterLily. It evaluates the control point pathing functions at the time step `t_frame`.

    - `cp_funcs` = The control point pathing array.
    - `Ncps` = The number of control points. Required for static array construction.
    - `t_frame` = The chosen time step.
"""
cps_at_time(cp_funcs, Ncps, t_frame) = SMatrix{2,Ncps,Float64}(hcat([f(t_frame) for f in cp_funcs]...) )

"""
The duty cycle interpolator function. This function is used to upsample the number of frames that define the motion cycle and implement the user-defined duty cycle into the motion cycle. The duty cycle is defined as the contraction period divided by the total period. The original case has a duty cycle `γ=0.4`, but this can be adjusted by the user. Based on this, the number of upsamples for respectively the contraction phase and expansion phase is adjusted. By plotting it on the same time grid (of 1 period), it will implement the duty cycle in the kinematics. It outputs a new sequence of control point sets that define 1 motion cycle. 
    
    - `cps_seq` = The sequence of control point sets in vector form.
    - `n_contr::Int` = Number of contraction frames, originally 4.
    - `n_exp::Int` = Number of expansion frames, originally 6.
    - `base_up::Int=10` = Number of upsamples, originally 10.
    - `γ::Real=1` = The user-defined duty cycle.

"""
function interpolate_cycle_duty(cps_seq; n_contr::Int, n_exp::Int, base_up::Int=10, γ::Real=1)
    Nframes = length(cps_seq)
    Npts = size(cps_seq[1], 2)
    @assert Nframes == n_contr + n_exp "length(cps_seq) must be n_contr + n_exp"

    xs = [cps_seq[t][1,j] for t in 1:Nframes, j in 1:Npts]
    ys = [cps_seq[t][2,j] for t in 1:Nframes, j in 1:Npts]

    # itp_x = [interpolate(xs[:,j], BSpline(Linear())) for j in 1:Npts]
    # itp_y = [interpolate(ys[:,j], BSpline(Linear())) for j in 1:Npts]

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

"""
This functions transforms the control point arrays into interpolation functions that define the control point pathing using a 1D Splines approach. 

    - `sx` = input of the x-pathing array.
    - `sy` = input of the y-pathing array.
    - `t_points` = the according time points of the given pathing arrays.
"""
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

"""
This is the algorithm to generate the array with (interpolation) functions for individual control point coordinates. The result is a `2xNcps` array with interpolation functions dependent on time. The first row are the x-coordinate pathings and the second row are the y-coordinate pathings. It applies the geometry, optimiser and kinematics algorithms altogether to generate the full motion of the jellyfish that should be simulated. So it defines the motion/kinematics over the full duration, defined as `duration = n_cycles * period`. It starts with refining the polyline, discretising it into a user-defined number of control points `Ncps`, for the contraction and expansion phase. Then, it optimises the control point positions to acquire mass conservation, using the first frame as the reference frame. Then the contraction and expansion frames are concatenated into a vector. This is a sequence of 10 control point sets, which is the input for duty cycle implementation. The result is 1 fully defined motion cycle, with the user-defined duty cycle implemented.

Varying period and gliding intervals are then added by creating a second motion cycle based on the period fraction `T2/T1` or repeating the first frame a number of times to acquire `Tg`, respectively. This is repeated for the user-defined number of cycles. The control points are than reversed (a requirement to have the flow outside of the jellyfish). Individual control point pathings are then computed and smoothened. The smoothened pathings are then projected/interpolated on the time point array and the result is there.

Best to continuously assign Static Arrays including size and type to avoid erros. (SMatrix{rows,columns,type}())

    - `contr_cps` = sequence of control point sets belonging to the contraction phase.
    - `exp_cps` = sequence of control point sets belonging to the expansion phase.
    - `Ncps` = number of control points.
    - `T1` = non-dimensional period length of the original, first period.
    - `T2` = non-dimensional period length of the consecutive period.
    - `Tg` = non-dimensional period length of the gliding interval.
    - `n_cycles` = number of cycles that need to be simulated.
    - `n_up` = number of upsamples that are added in between the original frames.
    - `γ` = user-defined duty cycle.
    - `ThreeD=false` = setting to generate either a 2D or 3D jellyfish. Important for mass conservation purposes.
    - `varyingT=false` = should be `true` for cases where the consecutive period is varied.
    - `gliding=false` = should be `true` for cases with gliding intervals.

"""
function generate_jelly_motion(contr_cps, exp_cps, Ncps, T1, T2, Tg, n_cycles, n_up, γ; ThreeD=false, varyingT=false, gliding=false)
    phase_contr     = Vector{SMatrix{2,Ncps,Float64,2Ncps}}([refine_polyline(cps, Ncps) for cps in contr_cps])
    phase_exp       = Vector{SMatrix{2,Ncps,Float64,2Ncps}}([refine_polyline(cps, Ncps) for cps in exp_cps])

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
        period_frames_T2    = interpolate_cycle_duty(cps_seq; n_contr=4, n_exp=6, base_up=round(Int,(T2/T1)*n_up), γ=γ)
        period_frames       = vcat(period_frames_T1, period_frames_T2)
        frames              = repeat(period_frames, n_cycles)
    elseif varyingT == false && gliding == true
        period_frames       = vcat(period_frames_T1, gliding_frames)
        frames              = repeat(period_frames, n_cycles)
    elseif varyingT == true && gliding == true
        period_frames_T2    = interpolate_cycle_duty(cps_seq; n_contr=4, n_exp=6, base_up=round(Int,(T2/T1)*n_up), γ=γ)
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
    # sx_smooth = sx 
    # sy_smooth = sy


    if varyingT == false && gliding == false
        time_set    = range(0,T1 * n_cycles, length=length(sx_smooth[1]))
    elseif varyingT == true && gliding == false
        time_set    = range(0,(T1+T2) * n_cycles,length=length(sx_smooth[1]))   
    elseif varyingT == false && gliding == true
        time_set    = range(0, (T1+Tg) * n_cycles, length=length(sx_smooth[1]))
    elseif varyingT == true && gliding == true
        time_set    = range(0,(T1+T2+2Tg) * n_cycles,length=length(sx_smooth[1]))   
    end

    pathing         = control_point_functions(sx_smooth, sy_smooth, time_set)
    return pathing
end

Base.@kwdef struct ValidationData{T}
    t::Vector{T}
    vol_mc::Vector{T}
    vol_cav::Vector{T}
    velar_diam::Vector{T}
    FR::Vector{T}
    height::Vector{T}
    width::Vector{T}
end

function compute_validation_data(pathing, geom, kin; dt=0.01)
    timeran = collect(0:dt:4*kin.T1)
    n = length(timeran)

    vols    = Vector{Float64}(undef, n)
    diams   = Vector{Float64}(undef, n)
    mc_vols = Vector{Float64}(undef, n)
    FRs     = Vector{Float64}(undef, n)
    hs      = Vector{Float64}(undef, n)
    ds      = Vector{Float64}(undef, n)

    for (i, t) in enumerate(timeran)
        cps0    = SMatrix{2,geom.Ncps,Float64}(cps_at_time(pathing, geom.Ncps, t))
        cps_vol = SMatrix{2,15,Float64}(cps0[:, 1:15])

        h       = cps0[1,16]
        d       = 2 * maximum(cps0[2,:])
        FR      = h / d
        vol     = shape_volume(cps_vol)
        mc_vol  = shape_volume(cps0)
        diam    = 2 * cps0[2,15]

        vols[i]    = vol
        diams[i]   = diam
        mc_vols[i] = mc_vol
        FRs[i]     = FR
        hs[i]      = h
        ds[i]      = d
    end

    return ValidationData(
        t = timeran,
        vol_mc = mc_vols,
        vol_cav = vols,
        velar_diam = diams,
        FR = FRs,
        height = hs,
        width = ds
    )
end