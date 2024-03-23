using WaterLily
#using GLMakie
#GLMakie.activate!()
using Plots#; gr()
using StaticArrays
using ParametricBodies

#include("/home/alidtke/git/WaterLily.jl/examples/TwoD_plots.jl")
#include("./src/diagnostics.jl")

function fig(store)
    store_matrix = reduce(vcat,store')
    p1 = plot(store_matrix[:,4],store_matrix[:,5],
            linez=cumsum(store_matrix[:,1]),
            colorbar=:true, aspect_ratio=:equal,
            label="position")
    p2 = plot(store_matrix[:,2],store_matrix[:,3],
            linez=cumsum(store_matrix[:,1]),
            colorbar=:false, aspect_ratio=:equal,
            label="acceleration")
    p3 = plot(store_matrix[:,6],store_matrix[:,7],
            linez=cumsum(store_matrix[:,1]),
            colorbar=:false, aspect_ratio=:equal,
            label="velocity")
    plot(p1, p2, p3, layout= @layout[a{0.5w} [grid(2,1)]])
end

let 
    WaterLily.CFL(a) = WaterLily.CFL(a,Δt_max=0.2)
    # parameters
    Re = 250
    U = 1
    L = 2^5
    Λ = 2.0
    radius, center = L/2, SA[2L,L]
    duration = 10.0
    step = 0.2
    Fn = 0.5

    # fsi parameters
    mₐ = π*radius^2 # added-mass coefficent circle
    m = 1mₐ
    vel = SA[0.,0.]
    a0 = SA[0.,0.]
    pos = SA[0.,0.]
    g = SA[0.,-U^2/(Fn^2*L)]
    t_init = 0

    # rotation variables
    Im = 0.025*m*(radius^2+radius^2/Λ^2)
    Iₐ = (radius^2+radius^2/Λ^2)^2
    dω = 0.0; dω₀=0.0; ω = 0.0; rot = 0.0
    α = π/4

    # @TODO add rotational velocity in the velocity
    function map(x,t)
        SA[cos(α+rot) sin(α+rot); -sin(α+rot) cos(α+rot)]*(x-center)
    end

    # make a body
    body = AutoBody((x,t)->√sum(abs2, SA[x[1]/Λ,x[2]])-radius/Λ,map)

    # generate sim using a moving reference frame
    Ut(i,t::T) where T = convert(T,-vel[i])
    sim = Simulation((4L,4L), Ut, radius; ν=U*radius/Re, U, body)

    # get start time
    t₀ = round(sim_time(sim)); 
    global store=[]

    anim = @animate for tᵢ in range(t₀, t₀+duration; step=step)
#    @time @gif for tᵢ in range(t₀,t₀+duration;step)
#    @time for tᵢ in range(t₀,t₀+duration;step)

        # update
        t = sum(sim.flow.Δt[1:end-1])
        while t < tᵢ*sim.L/sim.U

            # measure body
            measure!(sim,t)

            # update flow
            mom_step!(sim.flow,sim.pois);
            
            # pressure force
            force = -WaterLily.∮nds(sim.flow.p, sim.flow.f, sim.body, t)
            moment = 0.0
            # force = -WaterLily.∮nds(sim.flow.p,sim.flow.f,circle,t)
            #force,moment = diagnostics(sim,center)

            # compute motion and acceleration 1DOF
            Δt = sim.flow.Δt[end]
            accel = (force + m.*g + mₐ.*a0)/(m + mₐ)
            pos = pos + Δt.*(vel+Δt.*accel./2.) 
            vel = vel + Δt.*accel; 
            a0 = copy(accel)

            dω = (moment + dω₀*Iₐ)/(Im+Iₐ) 
            rot =+ Δt.*(ω+Δt*dω/2.)
            ω += Δt*dω; dω₀ = dω
            # @show dω ω rot

            # save dt, t, position, velocity, acceleration
            push!(store,[Δt, t, pos..., vel..., accel...])
            
            # update time, must be done globaly to set the pos/vel correctly
            t_init = t; t += Δt
        end

        # plot vorticity
        @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
#        flood(sim.flow.σ;clims=(-10,10)); body_plot!(sim)
        contourf(clamp.(sim.flow.σ, -10, 10)',
             color=palette(:RdBu_11), clims=(-10, 10), linewidth=0,
             aspect_ratio=:equal, legend=false, border=:none)
        #plot!(sim.body.surf)
        
        WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
        contour!(sim.flow.σ[inside(sim.flow.p)]'|>Array;levels=[0],lines=:black)
    
        plot!([center[1]],[center[2]],marker=:o,color=:red,legend=:none)

        # print time step
        println("tU/L=",round(tᵢ,digits=4),", Δt=",round(sim.flow.Δt[end],digits=3))
    end

    gif(anim, "outputs/plot_05_test_motion_ellipse.gif", fps=10)
end

# Convert to a column matrix for plotting
store = hcat(store...)'

plot(store[:, 2], store[:, 3], label="x", xlabel="Time", ylabel="Position")
plot!(store[:, 2], store[:, 4], label="y")
savefig("outputs/plot_06_test_motion_ellipse_pos.png")


