using StaticArrays, Plots, WaterLily, ParametricBodies, Interpolations, LinearAlgebra, Dierckx, GLMakie, Images, ImageMagick, ImageIO, BiotSavartBCs, DelimitedFiles, DataFrames, CSV, Statistics, WriteVTK, CairoMakie, Printf, Meshing, LaTeXStrings
include(joinpath(@__DIR__, "..", "data", "CP_arrays", "Sarsia_tubulosa.jl"))
include(joinpath(@__DIR__, "..", "data", "validation_arrays", "Sarsia_validation_arrays.jl"))
include("background_functions.jl")
include("geometry_algorithms.jl")
include("optimiser_algorithms.jl")
include("kinematics_algorithms.jl")
# include("simulation_functions_2D.jl")
include("simulation_functions_3D.jl")
include("visualisation_algorithms.jl")

""" --- Parameters --- 
Right now Tp and T are used as Float64 definition, but this should be changed to directly use Float64 at each location. 
Four sets of parameter structures are defined. One for simulation parameters, geometry parameters, kinematic parameters and scaling parameters.
"""

Tp = Float64
T = Float64

Base.@kwdef struct SimParams
    Re::Float64     = 302.0
    D::Int          = 2^6
    ϵ::Int          = 2                  # BDIM interface thickness, this means the thickness is 1 grid cell (which is standard in examples)
    deg::Int        = 1                # Polynomial degree to describe jellyfish boundary
    Uff::Float64    = 0.0          # 'far-field' velocity
    U::Float64      = 1.0            # Reference velocity
end

Base.@kwdef struct GeomParams
    Ncps::Int       = 35              # Number of control points to define jellyfish body
    n_up::Int       = 20              # General number of upsamples
    n_cycles::Int   = 50          # Number of cycles to be simulated.
end

Base.@kwdef struct KinParams
    T1::Float64     = 1.0
    T2::Float64     = 1.0
    Tg::Float64     = 1.0 
    γ::Float64      = 0.3            # Contraction phase/motion cycle ratio
end

Base.@kwdef struct ScalingParams
    Uavg::Float64   = 2.42
    Dmax::Float64   = 1.25
    tscale::Float64 = Uavg/Dmax
    St::Float64     = Dmax/Uavg         # Strouhal number (From Sahin 2009) (D/(UT)) based on avg medusa velocity of 2.42 cm/s
    Re::Float64     = 302               # Reynolds Number (From Sahin 2009) (UD/ν) based on avg medusa velocity of 2.42 cm/s
end

scaling::ScalingParams      = ScalingParams()
num::SimParams              = SimParams()
geom::GeomParams            = GeomParams()
kin::KinParams              = KinParams(T1 = 1*scaling.tscale, T2 = 1*scaling.tscale, Tg = 1*scaling.tscale)

"""
Define the contraction and expansion frames and pass them into the control point matrix generator.
"""

contr_frames                = [cps_0, cps_1, cps_2, cps_3] ./ scaling.Dmax              # 0, T/10, 2T/10, 3T/10, 4T/10 -> Required for correct upsampling.
exp_frames                  = [cps_4, cps_5, cps_6, cps_7, cps_8, cps_9] ./ scaling.Dmax        # 4T/10, 5T/10, 6T/10, 7T/10, 8T/10, 9T/10
# pathing                     = generate_jelly_motion(contr_frames, exp_frames, geom.Ncps, kin.T1, kin.T2, kin.Tg, geom.n_cycles, geom.n_up, kin.γ; ThreeD=false, varyingT=false, gliding=false)
pathing                   = generate_jelly_motion(contr_frames, exp_frames, geom.Ncps, kin.T1, kin.T2, kin.Tg, geom.n_cycles, geom.n_up, kin.γ; ThreeD=true, varyingT=false, gliding=false)
pathing2                  = generate_jelly_motion(contr_frames, exp_frames, geom.Ncps, kin.T1, kin.T2, kin.Tg, geom.n_cycles, geom.n_up, kin.γ; ThreeD=true, varyingT=false, gliding=false)
pathing3                  = generate_jelly_motion(contr_frames, exp_frames, geom.Ncps, kin.T1, kin.T2, kin.Tg, geom.n_cycles, geom.n_up, kin.γ; ThreeD=true, varyingT=false, gliding=false)



for t in 0:0.1:2
    plt = Plots.plot(cps_at_time(pathing_3D, geom.Ncps, t)[1,:], cps_at_time(pathing_3D, geom.Ncps, t)[2,:])
    Plots.scatter!(cps_at_time(pathing_3D, geom.Ncps, t)[1,:], cps_at_time(pathing_3D, geom.Ncps, t)[2,:])
    savefig(plt, "figures/2D3Dcheck/threeD_$t.png")
end

"""
Kinematic settings validation checks.
1. Check the pathing of a specific control point, preferably CP15 (assuming the initial settings) for the outer velar CP. (Maybe automate with a variable later)
2. Plot the geometry at a specific time step, using the cps_at_time function. Adjust `0` to a different time to check several time steps.
3. Generate the kinematic check plots on top of the Figures from Sahin 2009.
4. Check the signed distance field (deprecated function at this moment)
5. Check the grid size on the velum (deprecated function at this moment)
"""

Plots.plot(collect(0:0.01:4) ./ kin.T1, pathing[15](collect(0:0.01:4))[1], xlabel="time step", ylabel="x-coordinate", title="x-pathing of velar CP", label = "smoothened")                      
Plots.plot!(collect(0:0.01:4) ./ kin.T1, pathing2[15](collect(0:0.01:4))[1], xlabel="time step", ylabel="x-coordinate", title="x-pathing of velar CP", label = "no periodic s")                      
Plots.plot!(collect(0:0.01:4) ./ kin.T1, pathing3[15](collect(0:0.01:4))[1], xlabel="time step", ylabel="x-coordinate", title="x-pathing of velar CP", label = "no smoothing")                      


savepath        = "figures/validation/CP_continuity.pdf"
fig             = Figure(size=(figwidth_pt, figheight_pt))
ax              = jelly_axis(fig, doc_fontsize_pt;                   # Adjust axis labels
xlabel          = L"$\phi$ [-]",
ylabel          = L"$x_{15}(t)$ [-]"
)

# CairoMakie.xlims!(ax, 15, 20)                         # Axis limits
# CairoMakie.ylims!(ax, -1.5, -0.5)
l3 = CairoMakie.lines!(ax,
    collect(0:0.01:4) ./ kin.T1, pathing3[15](collect(0:0.01:4))[1];
    linewidth   = 1.2,
    color       = 5, colormap = cmap, colorrange = cr,
    linestyle   = linestyles[mod1(1, length(linestyles))],
    label       = "Linear pathings"
)

l1 = CairoMakie.lines!(ax,
    collect(0:0.01:4) ./ kin.T1, pathing[15](collect(0:0.01:4))[1];
    linewidth   = 1.2,
    color       = 1, colormap = cmap, colorrange = cr,
    linestyle   = linestyles[mod1(3, length(linestyles))],
    label       = "Smooth"
)

l2 = CairoMakie.lines!(ax,
    collect(0:0.01:4) ./ kin.T1, pathing2[15](collect(0:0.01:4))[1];
    linewidth   = 1.2,
    color       = 2, colormap = cmap, colorrange = cr,
    linestyle   = linestyles[mod1(2, length(linestyles))],
    label       = "No smoothing"
)



leg = Legend(fig, ax;
    orientation  = :horizontal,
    framevisible =  false,                          # Turn on/off of frame around legend
    labelsize    = doc_fontsize_pt                  # Label size the same as document Font
)

fig[2, 1] = leg                                     # Put legend in a new row below the axis

rowgap!(fig.layout, 2)                              # Adjust the gap between the legend and the figure.
rowsize!(fig.layout, 1, Relative(1))                # plot takes most space
rowsize!(fig.layout, 2, Relative(0.08))             # legend ~8% of height

save(savepath, fig; pt_per_unit=1)

Plots.plot(cps_at_time(pathing, geom.Ncps, 0)[1,:], cps_at_time(pathing, geom.Ncps, 0)[2,:])
Plots.scatter!(cps_at_time(pathing, geom.Ncps, 0)[1,:], cps_at_time(pathing, geom.Ncps, 0)[2,:])

generate_kin_checks(pathing, kin.T1, geom.Ncps; contr=true)
# signed_distance_field(pathing, num.deg, num.D, num.Re, num.U, num.D, num.Uff, kin.T1; tstart=0, tend=kin.T1, step=0.1)
# gridsize_on_flap(pathing, geom.Ncps, num.D)

"""
Generate the kinematic behaviour of the jellyfish over a time range. It can be used to compute:
(volume, oral cavity volume, bell width and length, fineness ratio, velar opening diameter and tip velocity)
All in structure form:
    t::Vector{T}
    vol_mc::Vector{T}
    vol_cav::Vector{T}
    velar_diam::Vector{T}
    FR::Vector{T}
    height::Vector{T}
    width::Vector{T}

Can then plot the data by calling the structure values.
"""

valdat1 = compute_validation_data(pathing, geom, kin; dt=0.01)
valdat2 = compute_validation_data(pathing_3D, geom, kin; dt=0.01)
Plots.plot(valdat1.t, valdat1.vol_cav)
Plots.plot!(valdat2.t, valdat2.vol_cav)

""" --- Simulation Setup 2 ---
In this part the actual simulations can be conducted.
Enter the data structures, set a simulation duration in terms of motion cycles and put pressure solver results plotting on or off.
First simulation is 2D, the second is 3D, choose the appropriate one and comment the other.

For visualisations, please check within the actual simulation functions. Normally, the visualisations are commented out. So to have them functional, turn them on and visualisation will occur and the results will be generated in the according folders.
A live-written .CSV file is stored in data/simulation_data, for which you can adjust the name as you like.
"""
duration = 15 # cycles

#2D
WaterLily.logger("data/test_psolver")
run_jelly_simulation(pathing_3D, duration, num, geom, kin)
plot_logger("data/test_psolver")
savefig("Figures/psolver.png")

#3D
WaterLily.logger("Data/Simulation_Data/test_psolver")
jelly_simulation_3D(pathing, duration, num, geom, kin)
plot_logger("Data/Simulation_Data/test_psolver")
savefig("Figures/psolver.png")

"""
--- Postprocessing ---
Create a gif from a folder filled with frames of either velocity, pressure or vorticity -> Change to VTK.
Other postprocessing can be done in the plotting routine files at this moment. -> Wrap these in functions and add here to simply plot whichever is required.
"""

function create_gif_from_folder(folder_path::String, output_path::String; delay::Float64=0.1, base_fps::Int=8)
    image_files = sort(filter(f -> any(ext -> endswith(lowercase(f), ext), [".png", ".jpg", ".jpeg"]),
                              readdir(folder_path, join=true)))

    function extract_float(path)
        m = match(r"([0-9]+(?:\.[0-9]+)?)(?=\.\w+$)", basename(path))
        return m === nothing ? Inf : parse(Float64, m.captures[1])
    end

    sorted_files = sort(image_files, by=extract_float)
    frames = [load(f) for f in sorted_files]

    repeats = max(1, round(Int, delay * base_fps))
    expanded_frames = reduce(vcat, [[frame for _ in 1:repeats] for frame in frames])

    save(output_path, cat(expanded_frames...; dims=3); fps=base_fps)
    println("GIF saved to: $output_path")
end
create_gif_from_folder("C:/Users/evanv/Desktop/Visuals/snapshots base case/Pinkjellyvisual/For presentation/Longerdomain", "figures/3Dlongerdomainslower2.gif", delay=0.15*(1.25/2.42))


rev_map(x,t)    = SA[(x[1]), hypot(x[2], x[3])] 
cps             =   cps_at_time(pathing, geom.Ncps, 0;) .* num.D .+ SA{Float64}[0.5num.D; 0]
weights         =   ones(Float64, size(cps, 2)); knots = Float64.(knots_vector(num.deg, size(cps, 2))); curve = NurbsCurve(cps, knots, weights )
body            =   ParametricBody(curve; map=rev_map, ndims=3)
sim             =   BiotSimulation((3num.D, num.D, num.D), (num.Uff,num.Uff,num.Uff), num.D; num.U, ν=(num.U*num.D) / num.Re, body, T, mem=Array, num.ϵ, nonbiotfaces=(-2,-3))

a               = sim.flow.σ
d               = similar(a,size(inside(a))) |> Array; # one quadrant
md              = similar(d, (1,2,2).*size(d))

geom            = geom!(md,d,sim) |> Observable;
# ω               = ω!(md, d, sim) |> Observable

fig             = GLMakie.Figure()
ax              = GLMakie.Axis3(fig[1, 1], aspect = :data)
GLMakie.xlims!(ax,0,3*D)
GLMakie.ylims!(ax,0,D)
GLMakie.zlims!(ax,0,D)

# GLMakie.volume!(ax, ω;algorithm=:mip,transparency=true,alpha=0.45,colormap=:algae,colorrange=(1,10))
GLMakie.mesh!(ax, geom, alpha=0.6, color=:red)

fig

df = CSV.read("data/simulation_data/full_jellyfish_geometry.csv", DataFrame)
Plots.plot(df[:,2], df[:,4])