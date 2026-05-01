"""
Function to generate pressure plots, only for 2D. Should also be applicable to 3D, but the result is then a 2D slice of the geometry.
    
    - `sim` = The simulation data
    - `tᵢ` = The convective time of the simulation
    - `Domain` = The domain for which the pressure should be visualised. The same for x and y.
"""
function gen_p_plots(sim, tᵢ, Domain)
    save_dir_p = joinpath("Figures", "Pressure_check")
    isdir(save_dir_p) || mkpath(save_dir_p)

    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))

    p = Array(sim.flow.p)        
    σ = Array(sim.flow.σ)          

    p_masked = copy(p)              
    p_masked[σ .< 0] .= NaN

    Nx, Ny = size(p_masked)
    x = range(0, Domain; length = Nx) ./ sim.L
    y = range(0, Domain; length = Ny) ./ sim.L

    pressure_plot = Plots.heatmap(x, y, p_masked', aspect_ratio=1,
    xlims=(0, Domain/sim.L), ylims=(0, Domain/sim.L), c=:balance, clims=(-2, 2),
    xlabel="x", ylabel="y", title="Pressure Field")

    Plots.contour!(x,y,sim.flow.σ',levels=[0])   
    savefig(pressure_plot, joinpath(save_dir_p, "pressure_$(tᵢ).png"))
end

"""
Function to generate velocity plots, only for 2D. Should also be applicable to 3D, but the result is then a 2D slice of the geometry.
    
    - `sim` = The simulation data
    - `tᵢ` = The convective time of the simulation
    - `Domain` = The domain for which the pressure should be visualised. The same for x and y.
"""

function gen_u_plots(sim, tᵢ, Domain)
    save_dir_p = joinpath("Figures", "Velocity_check")
    isdir(save_dir_p) || mkpath(save_dir_p)

    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))

    u = Array(.√(sim.flow.u[:,:,1].^2 .+ sim.flow.u[:,:,2].^2))
    σ = Array(sim.flow.σ)    

    Nx, Ny = size(u)
    x = range(0, 3Domain; length = Nx)
    y = range(0, Domain; length = Ny)

    u_masked = copy(u)              
    u_masked[σ .< 0] .= NaN      

    pressure_plot = Plots.heatmap(x,y,u_masked', aspect_ratio=1,
    xlims=(0, 3Domain), ylims=(0, Domain), c=:balance, clims=(-2, 2),
    xlabel="x", ylabel="y", title="Velocity Field")

    Plots.contour!(x,y,sim.flow.σ',levels=[0])   
    savefig(pressure_plot, joinpath(save_dir_p, "velocity_$(tᵢ).png"))
end

"""
Function to generate normal plots, only for 2D. Should also be applicable to 3D, but the result is then a 2D slice of the geometry.
    
    - `sim` = The simulation data
    - `tᵢ` = The convective time of the simulation
"""
function gen_n_plots(sim, tᵢ)
    save_dir = joinpath("Figures", "Normals_check")
    isdir(save_dir) || mkpath(save_dir)

    x = range(0, 130; length=130)
    y = range(0, 130; length=130)
    nx, ny = length(x), length(y)

    # Arrays for plotting (only nonzero vectors)
    xs, ys, nxs, nys = Float64[], Float64[], Float64[], Float64[]
    p_masked = fill(NaN, nx, ny)
    for j in 1:ny, i in 1:nx
        nvec = WaterLily.nds(sim.body, SVector(x[i], y[j]), 0.0)
        if norm(nvec) > 1e-6       # skip zero (or near-zero) vectors
            push!(xs, x[i])
            push!(ys, y[j])
            push!(nxs, nvec[1])
            push!(nys, nvec[2])
            p_masked[i,j] = sim.flow.p[i,j]
        end
    end

    fig = Figure(resolution=(700,700))
    ax = Axis(fig[1,1], title="Surface Normals (nds)", aspect=DataAspect())

    arrows!(ax, xs, ys, nxs, nys, arrowsize=10, lengthscale=3, color=:blue)
    save(joinpath(save_dir, "nds_frame_$(tᵢ).png"))
end

"""
Function to generate vorticity plots, only for 2D. Should also be applicable to 3D, but the result is then a 2D slice of the geometry.
    
    - `sim` = The simulation data
    - `tᵢ` = The convective time of the simulation
    - `D` = The domain for which the pressure should be visualised. The same for x and y.
"""
function gen_ω_gif(sim, t, D)
    save_dir_ω = joinpath("figures", "Vorticity_check")
    isdir(save_dir_ω) || mkpath(save_dir_ω)
    @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L/sim.U
    @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I]) < 0.001, 0.0, sim.flow.σ[I])
    ω = Array(sim.flow.σ)

    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
    σ = Array(sim.flow.σ)
    ω_masked = copy(ω)
    ω_masked[σ .< 0] .= NaN
    # vorticity_plot = flood(sim.flow.σ[R] |> Array; clims=(-1, 1))
    vorticity_plot = WaterLily.flood(ω_masked,clims=(-1,1),
              cfill=:seismic,legend=false,border=:none, xlims=(0, 6D),ylims=(0, 2D),
              xlabel="x", ylabel="y", title="Vorticity at tU/D=$(round(t, digits=4))")

    vorticity_plot = Plots.contour!(sim.flow.σ',levels=[0])
    savefig(vorticity_plot, joinpath(save_dir_ω, "vorticity_$(t).png"))
end

"""
Function to generate divergence plots, only for 2D. Should also be applicable to 3D, but the result is then a 2D slice of the geometry.
    
    - `sim` = The simulation data
    - `tᵢ` = The convective time of the simulation
    - `R` = The cells for visualisation
"""
function gen_div_gif(sim, t, R)
    @inside sim.flow.σ[I] = WaterLily.div(I,sim.flow.u)
    @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I]) < 0.001, 0.0, sim.flow.σ[I])
    WaterLily.flood(sim.flow.σ[R] |> Array; clims=(-0.5, 0.5))
    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
    Plots.contour!(sim.flow.σ',levels=[-sim.ϵ,0,sim.ϵ], title="$(round(t, digits=4))")
end

"""
Function that can be used to plot the grid size on the velum of the jellyfish. Can be used to visually check how well-defined this part of the jellyfish is. This function might not be working anymore.
"""
function gridsize_on_flap(pathing, Ncps, D)
    x = range(0,3D;step=1); y = range(0,D;step=1)
    plt = Plots.plot(legend=false, title="Grid Size on flap", xlabel="x [grid cells]", ylabel="y [grid cells]")
    for xi in x
        Plots.plot!([xi, xi], [first(y), last(y)], color=:gray, lw=1)
    end
    for yi in y
        Plots.plot!([first(x), last(x)], [yi, yi], color=:gray, lw=1)
    end

    Plots.plot!(cps_at_time(pathing, 2*Ncps-1, 0)[1,:].*D,cps_at_time(pathing, 2*Ncps-1, 0)[2,:].*D, xlims=(0.85*D,1.25D), ylims=(0.15D,D/2), 
    lw=2, color=:red)

    display(plt)
end

"""
Visualise the signed distance field at a specific time step.
"""
function signed_distance_field(pathing, deg, D, Re, U, Domain, Uff, period; tstart=0, tend=period, step=0.1)
    save_dir = joinpath(pwd(), "Figures/", "Shape_and_SDF_Studies/")
    isdir(save_dir) || mkpath(save_dir)
    xloc = Domain / 6; yloc = Domain/4; Domain_y = Domain
    xs = range(0, Domain, step=1)
    ys = range(0, Domain_y, step=1)
    times = range(tstart, tend, step=step) 
    for t in times
        cps         =   cps_at_time(pathing, 2*Ncps-1, t;) .* D .+ SA{T}[xloc, yloc] 
        weights     =   ones(T, size(cps, 2)); knots = Float64.(knots_vector(deg, size(cps, 2))); curve = NurbsCurve(cps, knots, weights )
        body        =   DynamicNurbsBody(curve; thk=0, boundary=true)
        sim         =   BiotSimulation((Domain, Domain_y), (Uff,Uff), D; U, ν=(U*D)/Re, body, mem=Array)

        Z           =   [sdf(body, WaterLily.loc(0, CartesianIndex(x, y), eltype(sim.flow.σ)), t) for y in ys, x in xs]
        sdf_plot    =   Plots.heatmap(xs, ys, Z; color=:algae, xlims=(0,Domain/2),ylims=(0,Domain_y/2), aspect_ratio=1, title="Signed Distance Field $t with deg $deg")
        Plots.contour!(xs, ys, Z, linewidth=2, color=:leonardo, levels=[0])  # Contour where sdf=0
        Plots.plot!(cps[1,:] .+ 1.5, cps[2,:] .+ 1.5, color=:red)
        savefig(sdf_plot, joinpath(save_dir, "sdf_t=$(t)_deg=$(deg)_D=$(D).png"))
    end
end

"""
Function to turn a set of png-files into a GIF from an input folder.
    
    - `folder_path::String` = the path to the folder with the png-files
    - `output_path::String` = the output path for the GIF-files
"""
function create_gif_from_folder(folder_path::String, output_path::String; delay::Float64=0.1)
    image_files = sort(filter(f -> any(ext -> endswith(lowercase(f), ext), [".png", ".jpg", ".jpeg"]), readdir(folder_path, join=true)))

    function extract_float(path)
        m = match(r"([0-9]+(?:\.[0-9]+)?) (?= \.\w+$)"x, path)
        return m === nothing ? Inf : parse(Float64, m.captures[1])
    end
    sorted_files = sort(image_files, by=extract_float)

    frames = [load(f) for f in sorted_files]

    save(output_path, cat(frames...; dims=3); fps=1/delay)
    println("GIF saved to: $output_path")
end

"""
A function to generate kinematic checks between the visuals of Sahin 2009 and the resulting control points of this model.

    - `pathing` = the set of interpolation functions for the control points
    - `period` = the original period of Sahin 2009 in non-dimensional units
    - `Ncps` = number of control points
    - `contr=true` = `true` for contraction phase plots, `false` for expansion phase plots
"""
function generate_kin_checks(pathing, period, Ncps; contr=true)
    time_range      = range(0,3period, length=30)
    colors          = [:black, :red, :green, :blue, :black, :red, :green, :blue, :purple, :lightblue,:black, :red, :green, :blue, :black, :red, :green, :blue, :purple, :lightblue]

    if contr == true
        img             = load("figures/Sarsia_contraction_frames.png")
        img_makie       = reverse(permutedims(img, (2, 1)), dims=2) 
        plot_steps      = 1:4
    else
        img             = load("figures/Sarsia_expansion_frames.png")
        img_makie       = reverse(permutedims(img, (2, 1)), dims=2) 
        plot_steps      = 5:10
    end

    fig             = GLMakie.Figure(resolution=(900, 600))
    ax              = GLMakie.Axis(fig[1, 1], aspect=DataAspect(), xlabel="x", ylabel="y")

    GLMakie.xlims!(ax, 0, 1.4)
    GLMakie.ylims!(ax, 0, 1.0)

    for i in plot_steps
        empty!(ax) 

        GLMakie.image!(ax, 0..1.4, 0..1, img_makie)

        t = time_range[i]
        cps_sm = cps_at_time(pathing, Ncps, t) .* 1.25

        xs_sm = Float32.(cps_sm[1, :])
        ys_sm = Float32.(cps_sm[2, :])

        # GLMakie.lines!(ax, xs_sm, ys_sm; color=:darkred, linewidth=7, linestyle=:dash)
        # GLMakie.scatter!(ax, xs_sm, ys_sm; color=:transparent,strokewidth = 1.2, strokecolor = :darkred, marker = :circle,markersize  = 8,)

        GLMakie.save("validation/frames/frame_$(lpad(i,3,'0')).png", fig)
    end
end

"""
--- Efficiency metric functions ---
"""

function compute_COT(time_ar, force_ar, pos_ar, vel_ar, pathing; γ=0.4, period=(2.42/1.25), n_per=10, Ncps=35, D=2^6)
    start           = findfirst(>(n_per*period), time_ar)
    ending2         = findfirst(>((n_per+1)*period), time_ar) - 1
    inst_pows       = []
    for i in start:1:ending2
        dt              = time_ar[i+1] - time_ar[i]
        cps0            = SMatrix{2,Ncps}(cps_at_time(pathing, Ncps, time_ar[i]))
        cps1            = SMatrix{2,Ncps}(cps_at_time(pathing, Ncps, time_ar[i+1]))
        # cp_vel          = abs(mean((cps1 - cps0) / dt)) 
        tip_displ       = norm(cps1[:,15] - cps0[:,15])
        tip_vel         = abs(tip_displ/dt)
        f_inst          = abs((force_ar[i] + force_ar[i+1]) / 2) ./ (D)^2
        P_inst          = tip_vel * f_inst
        push!(inst_pows, P_inst)
    end
    # Work = sum(inst_pows * dt)
    Pmean           = mean(inst_pows) 
    # dx = abs(pos_ar[ending2]) - abs(pos_ar[start]) ./ D
    COT             = Pmean / -mean(vel_ar)
    @show period, COT
    # COT = Work / dx 
    return COT, inst_pows
end

function run_case(sim, cfg; geom=geom,
                  D=2^6)

    pathing = generate_jelly_motion(
        contr_frames, exp_frames,
        geom.Ncps, cfg.T1, cfg.T2, cfg.Tg,
        geom.n_cycles, geom.n_up,
        cfg.γ;
        ThreeD=true,
        varyingT=false,
        gliding=true
    )

    COT, Pinst = compute_COT(
        sim.time, sim.force, sim.pos, sim.vel,
        pathing;
        γ=cfg.γ,
        period=(cfg.T1 + cfg.T2 + cfg.Tg),
        n_per=2,
        Ncps=geom.Ncps,
        D=D
    )

    η_fr = mean(abs.(sim.force./D^2) .* abs.(sim.vel)) / mean(Pinst)
    v_mean = mean(sim.vel)
    v_peak = minimum(sim.vel)
    return (; COT, η_fr, Pinst, v_mean, v_peak)
end

"""
Function that defines the axis conditions that I used for my general plotting routines. Up to anyone to change according to personal preferences.
"""
function jelly_axis(fig, doc_fontsize_pt; xlabel, ylabel)
    CairoMakie.Axis(fig[1,1];
        xlabel=xlabel,
        ylabel=ylabel,
        xgridvisible=true,
        ygridvisible=true,
        xgridcolor=(:black, 0.08),
        ygridcolor=(:black, 0.08),
        xticksmirrored=true,
        yticksmirrored=true,
        xminorticksvisible=true,
        yminorticksvisible=true,
        xminorticks=IntervalsBetween(5), 
        yminorticks=IntervalsBetween(5),
        xtickalign=1,                       
        ytickalign=1,
        xlabelsize = doc_fontsize_pt,       # Label size, equal to the document font size
        ylabelsize = doc_fontsize_pt,
        xticklabelsize = doc_fontsize_pt - 1,
        yticklabelsize = doc_fontsize_pt - 1,
        xticksize = -4,                 # Major ticks size, minus means outward smh?
        yticksize = -4,
        xlabelpadding = 2,              # Add distance between label and axis
        ylabelpadding = 4,
        xticklabelpad = 5,              # Add distance between label and tick
        yticklabelpad = 5,
        xminorticksize = -3,            # Small ticks size, minus means inward
        yminorticksize = -3
    )
end

## General plotting
figwidth_pt     = 452.97                            # Latex \textwidth in pt from the log
aspect          = 0.62                              # Ratio of height over width
figheight_pt    = figwidth_pt * aspect              # Figure height
doc_fontsize_pt = 12.5                              # Latex font size
set_theme!(Theme(font = "TeX Gyre Pagella",   fontsize = doc_fontsize_pt,))
linestyles      = [:solid, :dash, :dashdot, :dot]   # Different linestyle per line
cr              = (1,5) #extrema(Ds)                # linear range for the mapping
cmap            = :redsblues                        # pick any Makie colormap symbol

# function signed_distance_field_mom(deg, D, Re, U, Uff, pathing; t=0)
#     xs = range(0, 3D, step=1)
#     ys = range(0, D, step=1)

#     cps         =   cps_at_time(pathing, Ncps, 0;) .* D .+ SA{T}[0.5D; 0]
#     weights     =   ones(T, size(cps, 2)); knots = Float64.(knots_vector(deg, size(cps, 2))); curve = NurbsCurve(cps, knots, weights )
#     body        =   DynamicNurbsBody(curve; thk=0, boundary=true)
#     sim         =   BiotSimulation((3D, D), (Uff,Uff), D; U, ν=(U*D)/Re, body, mem=Array)

#     Z           =   [sdf(body, WaterLily.loc(0, CartesianIndex(x, y), eltype(sim.flow.σ)), 0) for y in ys, x in xs]

#     sdf_plot    =   Plots.heatmap(xs, ys, Z; color=:algae, xlims=(0,3D),ylims=(0,D), aspect_ratio=1, title="Signed Distance Field with deg $deg")


#     savepath        = "Data/Figures/methodology/SDFfield.pdf"
#     figwidth_pt     = 452.97
#     aspect          = 0.62
#     figheight_pt    = figwidth_pt * aspect
#     doc_fontsize_pt = 12.5

#     # --- Plot ---
#     cmap = :redsblues
#     maxabs = maximum(abs, filter(isfinite, vec(Z)))
#     cr = (-maxabs, maxabs)  # robust colorrange

#     # more contrast near 0 (p < 1)
#     p = 0.5
#     scale = ReversibleScale(
#         x -> sign(x) * abs(x / maxabs)^p,          # forward (for colors)
#         y -> sign(y) * abs(y)^(1/p) * maxabs       # inverse (for colorbar ticks)
#     )

#     set_theme!(Theme(font = "TeX Gyre Pagella", fontsize = doc_fontsize_pt))
#     fig = Figure(size = (figwidth_pt, figheight_pt))
#     ax = CairoMakie.Axis(fig[1, 1];
#         xlabel = L"$\mathbf{x}[1]$ [-]",
#         ylabel = L"$\mathbf{x}[2]$ [-]",
#         aspect = DataAspect(),      # like aspect_ratio=1
#         limits = (0, 3D, 0, D),
#     )

#     hm = CairoMakie.heatmap!(ax, xs, ys, permutedims(Z);
#         colormap   = :redsblues,
#         colorrange = cr,
#         colorscale = scale,
#         interpolate = false,
#     )

#     cb = CairoMakie.Colorbar(fig[2, 1], hm;
#         vertical = false,
#         flipaxis = false,
#         tickalign = 1,

#         height = 12,                 # makes it visually “thicker”
#         spinewidth = 1,

#         ticks = [-100,100],
#         # tickformat = x -> string(Int(round.(x))),

#         label = L"$d(\mathbf{x})$ [-]",
#         labelsize = doc_fontsize_pt,
#         ticklabelsize = doc_fontsize_pt,
#         labelpadding = 3,
#     )

#     # rowsize!(fig.layout, 2, Fixed(45))   # gives the colorbar row enough room
#     # rowgap!(fig.layout, 8)

#     # cb = CairoMakie.Colorbar(fig[2, 1], hm;
#     #     vertical = false,
#     #     flipaxis = false,                 # ticks below
#     #     tickalign = 1,                    # ticks outward
#     #     label = L"$d(\mathbf{x},t=0)$ [-]",
#     #     labelsize = doc_fontsize_pt,
#     #     ticklabelsize = doc_fontsize_pt,
#     # )
#     rowgap!(fig.layout, 2)
#     rowsize!(fig.layout, 2, Relative(0.10))

#     # colgap!(fig.layout, 10)

#     save(savepath, fig; pt_per_unit = 1)
# end

# signed_distance_field_mom(1, 64, 302, 1, Uff, pathing;)