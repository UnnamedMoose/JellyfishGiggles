import BiotSavartBCs: interaction,image,symmetry

"""
Function to apply the symmetry conditions in 3D to the y=0 and z=0 planes.
"""
@inline function symmetry(ω,T,args...) # overwrite to add image influences
    T₁,sgn₁ = image(T,size(ω),-2)
    T₂,sgn₂ = image(T,size(ω),-3)
    T₁₂,_   = image(T₁,size(ω),-3)
    return interaction(ω,T,args...)+sgn₁*interaction(ω,T₁,args...)+
        sgn₂*(interaction(ω,T₂,args...)+sgn₁*interaction(ω,T₁₂,args...))
end

"""
Function to mirror data from the first quadrant to the other 3 quadrants. Adjust the matrix definitions to mirror the other ways around.
"""
function mirrorto!(a,b)
    nx, ny, nz = size(b)

    # Fill quadrants from original block b (never from a)
    @views a[:, ny+1:2ny, nz+1:2nz]   .= b                    # y+ , z+
    # @views a[:, 1:ny, 1:nz] .= b
    @views a[:, 1:ny, nz+1:2nz] .= b[:, ny:-1:1, :]     # y− , z+
    @views a[:, ny+1:2ny,   1:nz] .= b[:, :, nz:-1:1]   # y+ , z−
    @views a[:, 1:ny, 1:nz] .= b[:, ny:-1:1, nz:-1:1] # y− , z−

    return a
end

"""
Function to visualise the geometry by evaluating the signed distance function of the grid domain. Can be copied to the other quadrants with the `mirrorto!` function.

    `d` = data
    `md` = mirrorred data
    `sim` = simulation data
"""
function geom!(md,d,sim,t=WaterLily.time(sim))
    a = sim.flow.σ
    WaterLily.measure_sdf!(a,sim.body,t)
    copyto!(d,a[inside(a)]) # copy to CPU
    mirrorto!(md,d)         # mirror quadrant
    md = d
    alg = Meshing.MarchingCubes()
    ranges = range.((0, 0, 0), size(md))
    points, faces = Meshing.isosurface(md, alg, ranges...)
    p3f = Point3f.(points)
    gltriangles = GLMakie.GLTriangleFace.(faces)
    return GLMakie.normal_mesh(p3f, gltriangles)
end

"""
Function to visualise the vorticity and mirror it to other quadrants, can be turned off by commenting the `mirrorto!` function.
"""
function ω!(md,d,sim)
    a,dt = sim.flow.σ,sim.L/sim.U
    @inside a[I] = WaterLily.ω_mag(I,sim.flow.u)*dt
    copyto!(d,a[inside(a)]) # copy to CPU
    mirrorto!(md,d)         # mirror quadrant
    md = d
end

"""
Function to generate a 2D strip of the vorticity field in a 3D simulation.
"""
function gen_ω_gif_3D(sim, t, Domain)
    save_dir_ω = joinpath("Figures", "Vorticity_check")
    isdir(save_dir_ω) || mkpath(save_dir_ω)
    @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L/sim.U
    @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I]) < 0.001, 0.0, sim.flow.σ[I])
    ω = Array(sim.flow.σ)
    ω2 = ω[:,:,2]
    @show size(ω2), typeof(ω2)

    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
    σ2 = Array(sim.flow.σ)[:,:,2]
    ω_masked = copy(ω2)
    ω_masked[σ2 .< 0] .= NaN
    @show size(ω_masked), typeof(ω_masked)
    # vorticity_plot = flood(sim.flow.σ[R] |> Array; clims=(-1, 1))
    vorticity_plot = WaterLily.flood(ω_masked,clims=(-5,5),
              cfill=:seismic,legend=false,border=:none, xlims=(0, 3D),ylims=(0, D),
              xlabel="x", ylabel="y", title="Vorticity at tU/D=$(round(t, digits=4))")

    vorticity_plot = Plots.contour!(σ2',levels=[0])
    savefig(vorticity_plot, joinpath(save_dir_ω, "vorticity_$(t).png"))
end

"""
Function to start and run the full 3D simulation on the jellyfish geometry. The first part defines the geometry and simulation environment. The second part initialises the constants and opens a CSV file to continuously store results of each time step. Then the simulation is conducted and the body is updated each step with the `update!` function. Geometry CPS are evaluated at each time step. The flow is then updated with `biot_mom_step!` and the jellyfish velocity and acceleration are applied to the background flow. Based on the resulting pressure from the NS equations, the forces are computed and the dynamic parameters are updated. Additional enstrophy and plots can be turned on for visualisation purposes.

    - `duration` = how long the simulation should continue, choose preferebly some integer * n_cycles
    - `period` = the period of one motion cycle, choose the initial one. T1.
    - `D` = grid size
    - `Re` = Reynolds number input
    - `U` = reference velocity
    - `ϵ` = kernel width
    - `pathing` = matrix with interpolation functions of the control points
    - `Uff` = far field velocity for each direction
    - `Ncps` = number of control points
"""
function jelly_simulation_3D(pathing, duration, num, geom, kin)
    rev_map(x,t)    = SA[(x[1]), hypot(x[2], x[3])] 
    cps             =   cps_at_time(pathing, geom.Ncps, 0;) .* num.D .+ SA{Float64}[0.5num.D; 0]
    weights         =   ones(Float64, size(cps, 2)); knots = Float64.(knots_vector(num.deg, size(cps, 2))); curve = NurbsCurve(cps, knots, weights )
    body            =   ParametricBody(curve; map=rev_map, ndims=3)
    sim             =   BiotSimulation((3num.D, num.D, num.D), (num.Uff,num.Uff,num.Uff), num.D; num.U, ν=(num.U*num.D) / num.Re, body, T, mem=Array, num.ϵ, nonbiotfaces=(-2,-3))
    
    # a               = sim.flow.σ
    # d               = similar(a,size(inside(a))) |> Array; # one quadrant
    # md              = similar(d, (1,2,2).*size(d))

    # geom            = geom!(md,d,sim) |> Observable;
    # ω               = ω!(md, d, sim) |> Observable

    # fig             = GLMakie.Figure()
    # ax              = GLMakie.Axis3(fig[1, 1], aspect = :data)
    # GLMakie.xlims!(ax,0,3*D)
    # GLMakie.ylims!(ax,0,D)
    # GLMakie.zlims!(ax,0,D)

    # GLMakie.volume!(ax, ω;algorithm=:mip,transparency=true,alpha=0.45,colormap=:algae,colorrange=(1,10))
    # GLMakie.mesh!(ax, geom, alpha=0.6, color=:red)

    # fig

    t₀ = round(sim_time(sim)); step = 0.1; t0 = 0; a0 = 0; v0 = 0; p0 = 0; f0 = 0; τ = 0.03 * kin.T1 * sim.L/sim.U;    

    open("data/simulation_results/results_3D.csv", "w") do io
        println(io, "forces,time,acceleration,velocity,position,volume")
        for tᵢ in range(t₀, t₀ + (duration * kin.T1); step)        
            t = sum(sim.flow.Δt[1:end-1])
            while t < tᵢ * sim.L / sim.U
                cps             = cps_at_time(pathing, geom.Ncps, t* sim.U/sim.L) .* num.D .+ SA{T}[0.5num.D; 0]
                # r = maximum(cps[2,:]); h = maximum(cps[1,:]) - 0.5*num.D; α = (h / r)^1.4
                sim.sim.body    = ParametricBodies.update!(sim.sim.body, cps, sim.flow.Δt[end])

                measure!(sim)
                biot_mom_step!(sim.flow,sim.pois,sim.ω,sim.x₀,sim.tar,sim.ftar;
                               fmm=sim.fmm,udf=fall!,acceleration=SA[-a0,0.f0,0.f0],U=SA[-v0,0.f0,0.f0])

                force_can       =   -4*WaterLily.total_force(sim)[1]
                Δt              =   sim.flow.Δt[end]
                vol             =   shape_volume(cps)                       # non-dimensionally: V_nd = V/(D^3)
                α               =   1 - exp(-Δt / (τ))
                force           =   (1-α)*f0 + α *force_can
                accel           =   force / vol                             # non-dimensionally: a_nd = f/(U^2D^2) / V/(D^3) = (fD^3)/(VU^2D^2) = f/V * (D/U^2), D/U^2 and U^2/D = acceleration scaling so correct.
                p0              +=  Δt * (v0 + Δt * accel / 2.)
                v0              +=  Δt * accel                              # non-dimensionally: v_nd = (ΔtU/D) * (aD/U^2) = (Δta)/U, and again U = velocity scaling so correct.
                a0              =   accel
                f0              =   force
                tnum            =   t * sim.U/sim.L

                if !isfinite(force)
                    println("Diverging Solution")
                end

                @inside sim.flow.σ[I] = WaterLily.ω_mag(I,sim.flow.u)*sim.L/sim.U
                enstr = sum(Array(sim.flow.σ).^2)

                println(io, "$force,$tnum,$a0,$v0,$p0,$force_can")

                t0 = t; t += Δt
            end
            # update viz fields
            # geom[] = geom!(md, d, sim)
            # ω[]    = ω!(md, d, sim)

            # gen_ω_gif_3D(sim, tᵢ, num.D)

            # fn = @sprintf("Figures/frames/frame_%04d.png", Int(tᵢ*10))
            # save(fn, fig)
            
            println("tU/L=", round(tᵢ, digits = 4), ", Δt=", round(sim.flow.Δt[end], digits = 3))
        end 
    end
end
