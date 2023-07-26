# WaterLily and misc addons
using WaterLily
# Standard modules
using StaticArrays
using GLMakie
using CUDA

# Functions for retrieving data from the GPU
function get_omega!(vort,sim)
    @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u) * sim.L / sim.U
    copyto!(vort,sim.flow.σ[inside(sim.flow.σ)])
end
function get_body!(bod,sim,t=WaterLily.time(sim))
    @inside sim.flow.σ[I] = WaterLily.sdf(sim.body,SVector(Tuple(I).-0.5f0),t)
    copyto!(bod,sim.flow.σ[inside(sim.flow.σ)])
end

# Main plotting routine.
function body_omega_fig(sim,resolution=(1400,700))
    # Set up figure
    fig = Figure(;resolution)
    ax = Axis(fig[1, 1]; autolimitaspect=1)
    hidedecorations!(ax); hidespines!(ax)

    # Get first vorticity viz
    vort = sim.flow.σ[inside(sim.flow.σ)] |> Array; ovort = Observable(vort)
    get_omega!(vort,sim); notify(ovort)
    contourf!(ax, ovort, levels=range(-20, 20, 21), colormap=:bluesreds)
    
    # Set up body viz
    bod = sim.flow.σ[inside(sim.flow.σ)] |> Array; obod = Observable(bod)
    get_body!(bod,sim); notify(obod)
    colormap = to_colormap([:grey30,(:grey,0.5)])
    contourf!(ax,obod,levels=[-100,0,1];colormap)
    
    # Return the objects
    fig,(vort,ovort,bod,obod)
end

# Update data by retrieving values from the GPU.
function update!(viz,sim)
    vort,ovort,bod,obod = viz
    get_omega!(vort,sim); notify(ovort)
    get_body!(bod,sim); notify(obod)
end

# Routine for setting up the simulation.
function make_sim(; L=32, Re=1e3, U=1, n=8, m=4, T=Float32, mem=Array)
    nose = SA[L, 0.5f0m*L]
    θ = T(π/18)

    function map(x, t)
        # Rotation matrix for the constant angle of attack.
        R = SA[cos(θ) -sin(θ); sin(θ) cos(θ)]
        # move to origin and align with x-axis
        ξ = R*(x-nose)
        # Return the transformed coordinate.
        return ξ
    end
    
    # Signed distance function for a straight line segment.
    function sdf(ξ, t)
        # Vector closest point on line to ξ
        p = ξ - SA[clamp(ξ[1], 0, L), 0]
        # Distance with a thickness offset
        return p'*p - 2
    end
    body = AutoBody(sdf, map)

    return Simulation((n*L, m*L), (U, 0), L; ν=U*L/Re, body, T, mem)
end

# === main ===
@assert CUDA.functional()

Makie.inline!(false)

L = 32
name = "outputs/out_00_motion.mp4"

cycle = range(0, L*2, 31)
sim = make_sim(; L, mem=CuArray)
fig,viz = body_omega_fig(sim)

@time Makie.record(fig, name, cycle, framerate=10) do t
    println("Current time ", t)
    sim_step!(sim, t)
    update!(viz, sim)
end

