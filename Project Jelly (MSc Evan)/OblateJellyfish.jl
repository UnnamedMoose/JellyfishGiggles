using LinearAlgebra, Printf, Statistics, Plots, ParametricBodies, StaticArrays
using WaterLily, CUDA
using GeometryBasics, Optim
using BiotSavartBCs
using DelimitedFiles, DataFrames

include("JellyfishGeometry.jl")

include("Metrics.jl")

include("SimulationSetup.jl")

include("InterpolationFunctions.jl")

T = Float32

cps_0 = SA{T}[0.000000  0.016454  0.055977  0.158851  0.290532  0.388024  0.479825  0.588982  0.715589  0.865430 0.738776  0.623858  0.532199  0.452133  0.458274  0.522241  0.603397  0.655723  0.685028  0.685597;
              0.000000  0.201646  0.397531  0.599177  0.800823  0.910288  1.002469  1.077366  1.111934  1.100412 1.077366  1.002469  0.875720  0.731687  0.639506  0.495473  0.374486  0.259259  0.138272  0.000000]

cps_1 = SA{T}[0.000000  0.016454  0.055977  0.158851  0.290532  0.388024  0.497228  0.600718  0.727348  0.877143 0.750536  0.652783  0.549530  0.452133  0.464035  0.522241  0.603397  0.655723  0.685028  0.685597;
              0.000000  0.201646  0.397531  0.599177  0.800823  0.910288  0.973663  1.025514  1.054321  1.054321 1.019753  0.973663  0.864198  0.731687  0.639506  0.495473  0.374486  0.259259  0.138272  0.000000]

cps_2 = SA{T}[0.000000  0.016454  0.055977  0.158851  0.290532  0.399594  0.508845  0.623953  0.750678  0.900496 0.756606  0.676066  0.566909  0.452133  0.464035  0.522241  0.603397  0.655723  0.685028  0.685597;
              0.000000  0.201646  0.397531  0.599177  0.800823  0.898765  0.950617  0.979424  0.985185  0.979424 0.944856  0.916049  0.841152  0.731687  0.639506  0.495473  0.374486  0.259259  0.138272  0.000000]

cps_3 = SA{T}[0.000000  0.016454  0.055977  0.158851  0.290532  0.411187  0.520486  0.676018  0.779769  0.941252 0.785744  0.687849  0.584311  0.469441  0.464035  0.522241  0.603397  0.655723  0.685028  0.685597;
              0.000000  0.201646  0.397531  0.599177  0.800823  0.881481  0.921811  0.927572  0.916049  0.875720 0.864198  0.852675  0.812346  0.725926  0.639506  0.495473  0.374486  0.259259  0.138272  0.000000]

cps_4 = SA{T}[0.000000  0.016454  0.055977  0.158851  0.290532  0.411235  0.520558  0.676066  0.826073  0.941608 0.791671  0.693729  0.590120  0.486748  0.464035  0.522241  0.603397  0.655723  0.685028  0.685597;
              0.000000  0.201646  0.397531  0.599177  0.800823  0.869959  0.904527  0.916049  0.864198  0.789300 0.823868  0.823868  0.800823  0.720165  0.639506  0.495473  0.374486  0.259259  0.138272  0.000000]

cps_5 = SA{T}[0.000000  0.016454  0.055977  0.158851  0.290532  0.399675  0.514688  0.675886  0.825965  0.907026 0.785778  0.699263  0.578513  0.486735  0.469712  0.522241  0.603397  0.655723  0.685028  0.685597;
              0.000000  0.201646  0.397531  0.599177  0.800823  0.881481  0.933333  0.962140  0.893004  0.795062 0.858436  0.881481  0.823868  0.725926  0.662551  0.495473  0.374486  0.259259  0.138272  0.000000]

cps_6 = SA{T}[0.000000  0.016454  0.055977  0.158851  0.290532  0.399580  0.514451  0.675578  0.802540  0.883649 0.791231  0.681648  0.566801  0.480950  0.469712  0.522241  0.603397  0.655723  0.685028  0.685597;
              0.000000  0.201646  0.397531  0.599177  0.800823  0.904527  0.990947  1.037037  0.985185  0.875720 0.933333  0.962140  0.869959  0.731687  0.662551  0.495473  0.374486  0.259259  0.138272  0.000000]

cps_7 = SA{T}[0.000000  0.016454  0.055977  0.158851  0.290532  0.399556  0.514356  0.675436  0.790781  0.877461 0.767878  0.664245  0.555254  0.480950  0.469712  0.522241  0.603397  0.655723  0.685028  0.685597;
              0.000000  0.201646  0.397531  0.599177  0.800823  0.910288  1.013992  1.071605  1.042798  0.979424 1.008230  0.990947  0.875720  0.731687  0.662551  0.495473  0.374486  0.259259  0.138272  0.000000]

cps_8 = SA{T}[0.000000  0.016454  0.055977  0.158851  0.290532  0.399556  0.514356  0.675365  0.773307  0.877177 0.750428  0.652699  0.555254  0.486711  0.469712  0.522241  0.603397  0.655723  0.685028  0.685597;
              0.000000  0.201646  0.397531  0.599177  0.800823  0.910288  1.013992  1.088889  1.088889  1.048560 1.048560  0.996708  0.875720  0.731687  0.662551  0.495473  0.374486  0.259259  0.138272  0.000000]

cps_9 = SA{T}[0.000000  0.016454  0.055977  0.158851  0.290532  0.399556  0.514356  0.669509  0.773188  0.876987 0.744572  0.646914  0.549493  0.475189  0.469712  0.522241  0.603397  0.655723  0.685028  0.685597;
              0.000000  0.201646  0.397531  0.599177  0.800823  0.910288  1.013992  1.111934  1.117695  1.094650 1.071605  1.002469  0.875720  0.731687  0.662551  0.495473  0.374486  0.259259  0.138272  0.000000]

oblate_cps_set = [cps_0, cps_1, cps_2, cps_3, cps_4, cps_5, cps_6, cps_7, cps_8, cps_9]

function make_symmetric_jelly(cps_list::AbstractVector{<:SMatrix{2,N,T}};
                                        tol = nothing) where {N,T}
        tol === nothing && (tol = sqrt(eps(T)))

        first_cps = cps_list[1]::SMatrix{2,20,Float32,40}
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

oblate_cps_set = make_symmetric_jelly(oblate_cps_set)                   # Vector{Matrix{Float32}}

oblate_cps_set = reverse_cps_list(oblate_cps_set)                       # Change the cps_list from clockwise to counterclockwise order

oblate_curves = [BSplineCurve(cps; degree=1) for cps in oblate_cps_set]


degree = 4
n_ctrl = size(oblate_cps_set[1], 2)
weights = ones(T, n_ctrl)
knots = T.(clamped_uniform_knots(degree, n_ctrl))

oblate_curves = [NurbsCurve(cps, knots, weights) for cps in oblate_cps_set]

plt = plot(oblate_curves[1], color=:red, alpha=0.5, xlims=(0, 1.4), ylims=(0, 1.4), title="NURBS Curves", legend=:false)
    plot!(oblate_curves[2], color=:blue, alpha=0.5)
    plot!(oblate_curves[3], color=:green, alpha=0.5)
    plot!(oblate_curves[4], color=:orange, alpha=0.5)
    plot!(oblate_curves[5], color=:purple, alpha=0.5)

    display(plt)

plt2 = plot(oblate_curves[6], color=:red, alpha=0.5, xlims=(0, 1.4), ylims=(0, 1.4), title="NURBS Curves", legend=:false)
    plot!(oblate_curves[7], color=:blue, alpha=0.5)
    plot!(oblate_curves[8], color=:green, alpha=0.5)
    plot!(oblate_curves[9], color=:orange, alpha=0.5)
    plot!(oblate_curves[10], color=:purple, alpha=0.5)

    display(plt2)

@inline function dynamicSpline(::Type{T}=Float32; new_cps_list,D=2^7,Re=302,U=1,ϵ=0.5,thk=2ϵ+√3,mem=Array, use_biotsavart=false) where {T<:AbstractFloat}
    cps = new_cps_list[1] .* D .+ SA{T}[3D,4D]
    degree = 3
    n_ctrl = size(cps, 2)
    weights = ones(T, n_ctrl)
    knots = T.(clamped_uniform_knots(degree, n_ctrl))

    curve = NurbsCurve(cps, knots, weights)         

    body = DynamicNurbsBody(curve; thk=thk, boundary=true)
    ν = U*D/Re
    return use_biotsavart ?
    BiotSimulation((10D,8D),(0,0),D; U, ν, body, T, mem, ϵ) :
    Simulation((10D,8D),(0,0),D; U, ν, body, T, mem,ϵ, 
    # exitBC=true   
    )
end
new_cps_list = oblate_cps_set
D = 2^5; Re = 302; U = 1; ϵ = 0.5; thk = 1                                                    # Simulation parameters, D = number of grid cells over jelly diameter.
sim             = dynamicSpline(; new_cps_list, D, Re, U, ϵ, thk, mem=Array, use_biotsavart=true);
Tp              = eltype(sim.flow.p) 
periodic_force  = Tp(0); v = Tp(0); s = Tp(0); areas = Tp[]; τ_locals = Tp[]
period          = Tp(3) 

function get_body!(bod,sim,t=WaterLily.time(sim))
    @inside sim.flow.σ[I] = WaterLily.sdf(sim.body,SVector(Tuple(I).-0.5f0),t)
    copyto!(bod,sim.flow.σ[inside(sim.flow.σ)])
end

addbody(x,y;c=:black) = Plots.plot!(Shape(x,y), c=c, legend=false)
function body_plot!(sim;levels=[0],lines=:black,R=inside(sim.flow.p),title)
    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
    contour!(sim.flow.σ[R]'|>Array;levels,lines, title=title)        # Plot signed distance function of body
    plot!(sim.body.curve, shift=(0.5, 0.5), add_cp=true)
    # xs = range(0, 300, length=200)
    # ys = range(0, 300, length=200)
    # Z = [sdf(sim.body, SA[x, y]) for y in ys, x in xs]

    # heatmap(xs, ys, Z; color=:viridis, aspect_ratio=1, title="Signed Distance Field")
    # contour!(xs, ys, Z, levels=[0.0], linewidth=2, color=:green, title=title)  # Contour where sdf=0

    # heatmap(sim.flow.σ[R]', clim=(-0.1, 0.1), title=title)  # this shows small nonzero ghost blobs
end

function sim_gif_forces!(sim, new_cps_list;
                         duration=1, period=3, step=0.1, verbose=true,
                         R=inside(sim.flow.p), remeasure=false, plotbody=false, kv...)

    Tp = eltype(sim.flow.p)
    t₀ = round(sim_time(sim))
    t = sum(sim.flow.Δt[1:end-1])  # current sim time

    v = Float32(0); s = Float32(0)
    in_period = true
    periodic_force = zero(Tp)
    t_start = 0

    # --- storage for force history ---
    # ts   = Tp[]
    # dts  = Tp[]
    f_hist = Tp[]      # store Fx or full force vector if needed
    interpolated_shapes = []
    # step = []
    # interp_state = Tp[]
    # div = Tp[]
    period = period * sim.L / sim.U

    anim = @animate for tᵢ in range(t₀, t₀+duration; step)
        while t < tᵢ * sim.L / sim.U
            # adaptive timestep
            sim.flow.Δt[end] = WaterLily.CFL(sim.flow; Δt_max=Tp(0.1))
            # Δt = sim.flow.Δt[end]
            cps_interp, k, τ_total, state = interpolate_cps_hermite_new(new_cps_list, t, period)
            @show t, state
            # push!(step, k)

            body_interpolation = cps_interp .* sim.L .+ (Tp(3sim.L), Tp(4sim.L))
            # push!(interpolated_shapes, body_interpolation)
            # @show typeof(body_interpolation)
            # @show typeof(sim.sim.body)
            # body_interpolation = SMatrix{2, size(cps_k,2), Float32}(body_interpolation)

            sim.sim.body = ParametricBodies.update!(sim.sim.body, body_interpolation, sim.flow.Δt[end])

            # --- advance one step ---
            sim_step!(sim, tᵢ; remeasure)

            # verbose && @show scaled
            # push!(interp_state, t)

            # if in_period
            #     periodic_force += scaled[1]
            #     if t - t_start >= period
            #         in_period = false
            #     end
            # end
            # raw    = WaterLily.total_force(sim)
            # scaled = raw ./ (0.5 * sim.L * sim.U^2)
            # push!(f_hist, scaled[1])  # store x-force (or push!(f_hist, scaled) to keep both)
            # push!(ts, t)
            t += sim.flow.Δt[end]
        end
        # --- forces ---

        # push!(dts, sim.flow.Δt[end])

        # --- visualization ---
        @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L/sim.U
        @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I]) < 0.001, 0.0, sim.flow.σ[I])
        # @inside sim.flow.σ[I] = WaterLily.div(I,sim.flow.u)
        # @show maximum(abs,sim.flow.σ[R]|>Array)
        # push!(div, maximum(abs,sim.flow.σ[R]|>Array))
        flood(sim.flow.σ[R] |> Array; clims=(-5,5), kv...)
        # contour(sim.flow.p')
        plotbody && body_plot!(sim; title="$tᵢ")

        verbose && println("t=", round(t, digits=4),
                           ", Δt=", round(sim.flow.Δt[end], digits=3))
    end 

    gif(anim,"Swimming_Jelly.gif")

    return (#ts=ts, 
            # dts=dts,
            interpolated_shapes = interpolated_shapes, 
            forces=f_hist,
            #periodic_force=periodic_force,
            #interp_state=interp_state, 
            #div=div
            )
end

cycles          = 2
duration        = cycles * period


WaterLily.logger("test_psolver")
res             = sim_gif_forces!(sim, new_cps_list; duration, period, step = 0.1, remeasure = true, plotbody = true)
# show the pressure logger
plot_logger("test_psolver")
savefig("psolver.png")