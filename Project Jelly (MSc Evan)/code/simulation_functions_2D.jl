import BiotSavartBCs: interaction,image,symmetry

"""
Function to apply the symmetry conditions to the y=0 plane. Change the `-2` to apply symmetry conditions to other planes. This is for 2D.
"""
@inline function symmetry(ω,T,args...) # overwrite to add image influences
    T₁,sgn₁ = image(T,size(ω),-2)
    return interaction(ω,T,args...)+sgn₁*interaction(ω,T₁,args...)
end

"""
Function to start and run the full 2D simulation on the jellyfish geometry. The first part defines the geometry and simulation environment. The second part initialises the constants and opens a CSV file to continuously store results of each time step. Then the simulation is conducted and the body is updated each step with the `update!` function. Geometry CPS are evaluated at each time step. The flow is then updated with `biot_mom_step!` and the jellyfish velocity and acceleration are applied to the background flow. Based on the resulting pressure from the NS equations, the forces are computed and the dynamic parameters are updated. Additional enstrophy and plots can be turned on for visualisation purposes.

    - `duration` = how long the simulation should continue, choose preferebly some integer * n_cycles
    - `period` = the period of one motion cycle, choose the initial one. T1.
    - `D` = grid size
    - `pathing` = matrix with interpolation functions of the control points
    - `Ncps` = number of control points
    - `deg` = polynomial degree to define the geometry curve
"""
function run_jelly_simulation(pathing, duration, num, geom, kin)
    cps         =   cps_at_time(pathing, geom.Ncps, 0;) .* num.D .+ SA{T}[0.5num.D; 0] 
    weights     =   ones(T, size(cps, 2)); knots = Float64.(knots_vector(num.deg, size(cps, 2))); curve = NurbsCurve(cps, knots, weights)
    body        =   ParametricBody(curve; ndims=2)
    sim         =   BiotSimulation((3num.D, num.D), (num.Uff,num.Uff), num.D; num.U, ν=(num.U*num.D)/num.Re, body, T, mem=Array,num.ϵ, nonbiotfaces=(-2))

    t₀ = round(sim_time(sim)); step = 0.1; t0 = 0; a0 = 0; v0 = 0; p0 = 0; f0 = 0; τ = 0.03 * kin.T1 * sim.L/sim.U;    

    open("data/simulation_data/results_2D.csv", "w") do io
        println(io, "forces,time,velocity,acceleration,area")
        for tᵢ in range(t₀, t₀ + (duration * kin.T1); step)        
            t = sum(sim.flow.Δt[1:end-1])
            while t < tᵢ * sim.L / sim.U
                cps             = cps_at_time(pathing, geom.Ncps, t * sim.U/sim.L) .* num.D .+ SA{T}[0.5num.D; 0] 
                # r = maximum(cps[2,:]) - 0; h = maximum(cps[1,:]) - num.D/6; α = (h / r)^1.4
                sim.sim.body    = ParametricBodies.update!(sim.sim.body, cps, sim.flow.Δt[end])
                measure!(sim)
                biot_mom_step!(sim.flow,sim.pois,sim.ω,sim.x₀,sim.tar,sim.ftar;
                               fmm=sim.fmm,udf=fall!,acceleration=SA[-a0,0.f0],U=SA[-v0,0.0]) # change of frame
                if t < kin.T1 * sim.L/sim.U
                    force_can = 0
                else
                    force_can           =   -2*WaterLily.total_force(sim)[1]
                end
                Δt              =   sim.flow.Δt[end]
                area            =   shape_area(cps)
                α               =   0.20
                force           =   (1-α)*f0 + α *force_can
                accel           =   force / (area)
                # accel           =   (1-α)*a0 + α *a_can

                @show t, force, accel   
                p0              +=  Δt * (v0 + Δt * accel / 2.)
                v0              +=  Δt * accel
                a0              =   accel
                f0              =   force
                tnum            =   t * sim.U/sim.L

                if !isfinite(force)
                    println("Diverging Solution")
                end

                # @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L/sim.U
                # @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I]) < 0.001, 0.0, sim.flow.σ[I])
                # ω               =   Array(sim.flow.σ)
                # enstr           =   sum(ω.^2)

                println(io, "$force,$tnum,$v0,$a0,$area")

                t0 = t; t += Δt
            end

            # gen_p_plots(sim, tᵢ, num.D)
            # gen_u_plots(sim, tᵢ, num.D)
            gen_ω_gif(sim, tᵢ, num.D)
            # save!(wr, sim)
            
            println("tU/L=", round(tᵢ, digits = 4), ", Δt=", round(sim.flow.Δt[end], digits = 3))
        end 
    end
    # close(wr)
end







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
                # if t < period*sim.L / sim.U
                #     force = 0
                # else
                #     force           =   -WaterLily.total_force(sim)[1]
                # end