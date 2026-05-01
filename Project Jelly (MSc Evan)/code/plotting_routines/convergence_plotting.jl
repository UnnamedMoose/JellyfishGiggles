"""
--- GRID SIZE CONVERGENCE STATISTICS PLOTTING ---
"""

include(joinpath(@__DIR__, "..", "visualisation_algorithms.jl"))

function rmse_against_truth(sim, truth_time, truth_signal, norm)
    itp         = LinearInterpolation(sim.time, sim.force ./ (norm^2))
    s           = itp.(truth_time)
    return Float64(sqrt(mean((s .- truth_signal).^2)))
end

"""
1. Load the appropriate files
"""
#Load the grid convergence data
files               = [
    "results_3D_16.csv",
    "results_3D_32.csv",
    "results_3D_40.csv",
    "results_3D_48.csv",
    "results_3D_56.csv",
    "results_3D_64.csv",
    "results_3D_80.csv",
    "results_3D_96.csv",
    "results_3D_112.csv",
    "results_3D_128.csv",
    "results_3D_156.csv",
    "results_3D_256.csv"
    ]
sims                = load_simulation.("data/convergence_data/" .* files)
Ds                  = Float64[16,32,40,48,56,64,80,96,112,128,156,256]

#Load the domain convergence data
base                = CSV.read("data/duty_cycle_variation_data/3D_D64_eps2_dt005_basecase_gam040.csv", DataFrame)
twice               = CSV.read("data/convergence_data/results_3D_twice_the_domain.csv", DataFrame)

#Load the time step convergence data
# files               = [
#     "results_3D_D32_dt005.csv",
#     "results_3D_D32_dt010.csv",
#     "results_3D_D32_dt015.csv",
#     "results_3D_D32_dt020.csv",
#     "results_3D_D32_dt025.csv"
#     ]
files               = [
    "results_3D_dt005_32.csv",
    "results_3D_dt010_32.csv",
    "results_3D_dt015_32.csv",
    "results_3D_dt020_32.csv",
    "results_3D_dt025_32.csv"
    ]
dts_sims            = load_simulation.("data/convergence_data/" .* files)
dts                 = [0.05, 0.10, 0.15, 0.20, 0.25]

plt = Plots.plot(xlims=(1,5))
for f in files[1:5] 
    df = CSV.read("data/convergence_data/$f", DataFrame)
    Plots.plot!(df[:,3], df[:,2] ./ 32^2)
end
Plots.plot!(df3[:,2], 2 .* df3[:,1] ./ 64)
display(plt)

df1 = CSV.read("data/duty_cycle_variation_data/3D_D64_eps2_dt005_basecase_gam040.csv", DataFrame)
df2 = CSV.read("data/convergence_data/results_3D_dt010_64.csv", DataFrame)
df3 = CSV.read("data/convergence_data/results_3D_dt0010_64.csv", DataFrame)
"""
2. Compute the convergence statistics, root mean square error.
"""
# Grid size convergence statistics
rmses               = Float64[]
i1                  = findfirst(>(kin.T1), sims[end-1].time)
i2                  = findfirst(>(2*kin.T1), sims[end-1].time) - 1
t_truth             = sims[end-1].time[i1:i2]
s1                  = sims[end-1].force[i1:i2] ./ (Ds[end-1]^2)

for (i,sim) in enumerate(sims)
    rmse = rmse_against_truth(sim, t_truth, s1, Ds[i])
    @show rmse
    push!(rmses, rmse)
end

# Domain size convergence statistics
i1                  = findfirst(>(4kin.T1), base[:,2])
i2                  = findfirst(>(5kin.T1), base[:,2]) - 1
t_truth             = base[:,2][i1:i2]
s1                  = base[:,1][i1:i2] ./ (64^2)
itp                 = LinearInterpolation(twice[:,2], twice[:,1]./ (64^2))
s                   = itp.(t_truth)
ds_rmse             = sqrt(mean((s .- s1).^2))

# Time step convergence statistics
# struct SimulationData2
#     time   :: Vector{Float64}
#     force  :: Vector{Float64}
#     funfiltered :: Vector{Float64}
#     vel    :: Vector{Float64}
#     acc    :: Vector{Float64}
#     pos    :: Vector{Float64}
#     vol    :: Vector{Float64}
# end

# """
# Function to load a simulation CSV results file and acquire the data from it.
# """
# function load_simulation(file)
#     df = CSV.read(file, DataFrame)
#     return SimulationData2(
#         df.time,
#         df.forces,
#         df.funfiltered,
#         df.velocity,
#         df.acceleration,
#         df.position,
#         df.volume
#     )
# end


dt_rmses            = Float64[]
i1_dt               = findfirst(>(10*kin.T1), dts_sims[1].time)
i2_dt               = findfirst(>(11*kin.T1), dts_sims[1].time) - 1
t_truth_dt          = dts_sims[1].time[i1_dt:i2_dt]
s1_dt               = dts_sims[1].funfiltered[i1_dt:i2_dt] ./ (32^2)

for (i,sim) in enumerate(dts_sims)
    rmse = rmse_against_truth(sim, t_truth_dt, s1_dt, 32)
    push!(dt_rmses, rmse)
end


"""
3. Plot the convergence results. Either a convergence plot or a plot with the force signal over periods of time for varying numerical parameters.
"""
## Plot the grid size convergence
savepath        = "figures/convergence/grid_size_force_rmse_conv.pdf"
fig             = Figure(size=(figwidth_pt, figheight_pt))
ax              = jelly_axis(fig, doc_fontsize_pt;                   # Adjust axis labels
xlabel          = L"$D$ [-]",
ylabel          = L"rmse($f$) [-]"
)

CairoMakie.lines!(ax,
    Ds, rmses;
    linewidth   = 1.2,
    color       = 1,
    colormap    = cmap,
    colorrange  = cr,
    linestyle   = linestyles[mod1(1, length(linestyles))]
)

CairoMakie.scatter!(ax,
    Ds, rmses;
    marker      = :circle,
    markersize  = 8,
    color       = :transparent,
    strokecolor = cgrad(cmap)[1],     # ← same exact colour
    strokewidth = 1.2
)

save(savepath, fig; pt_per_unit=1)

## FIGURE OPTIONS AND PLOTTING ROUTINES
savepath        = "data/convergence/full_force_conv_plot.pdf"
fig             = Figure(size=(figwidth_pt, figheight_pt))
ax              = jelly_axis(fig, doc_fontsize_pt;                   # Adjust axis labels
xlabel          = L"$\phi$ [-]",
ylabel          = L"$f/(UD)^2$ [-]"
)
cr              = extrema(Ds)                 # linear range for the mapping
handles         = Makie.AbstractPlot[]   # store line objects
labels          = String[]          # store legend labels


CairoMakie.xlims!(ax, 0, 1)
# CairoMakie.ylims!(ax, -3.5, 2.5)

for (i, sim) in enumerate(sims)
    ln = CairoMakie.lines!(
        ax,
        sim.time ./ kin.T1 .- 10,
        sim.force ./ Ds[i]^2;
        linewidth=1.2,
        color=Ds[i],                 # <-- numeric value (gets colour-mapped)
        colormap=cmap,
        colorrange=cr,
        colorscale=identity,          # <-- linear (default, but explicit)
        linestyle=linestyles[mod1(i, length(linestyles))],
        # label=LaTeXString("\\gamma = $(cfg.γ)")
    )

    push!(handles, ln)
    push!(labels, "$(Ds[i])")
end

Legend(
    fig[1, 2],
    handles,
    labels;
    framevisible = false,
    tellheight   = false,
    labelsize    = doc_fontsize_pt,
    labelfont    = "TeX Gyre Pagella"
)

save(savepath, fig; pt_per_unit=1)

## DOMAIN SIZE CONVERGENCE STATISTICS PLOTTING
savepath        = "figures/convergence/domain_size_convergence.pdf"
fig             = Figure(size=(figwidth_pt, figheight_pt))
ax              = jelly_axis(fig, doc_fontsize_pt;                   # Adjust axis labels
xlabel          = L"Domain Size ($\cdot (3D,D,D)$) [-]",
ylabel          = L"rmse($f$) [-]"
)

CairoMakie.lines!(ax,
    [1,2], [1,1-ds_rmse];
    linewidth   = 1.2,
    color       = 1,
    colormap    = cmap,
    colorrange  = cr,
    linestyle   = linestyles[mod1(1, length(linestyles))]
)

CairoMakie.scatter!(ax,
    [1,2], [1,1-ds_rmse];
    marker      = :circle,
    markersize  = 8,
    color       = :transparent,
    strokecolor = cgrad(cmap)[1],     # ← same exact colour
    strokewidth = 1.2
)

save(savepath, fig; pt_per_unit=1)

## DOMAIN SIZE FULL SIGNAL CONVERGENCE
savepath        = "figures/convergence/domain_size_convergence_fullsignal.pdf"
fig             = Figure(size=(figwidth_pt, figheight_pt))
ax              = jelly_axis(fig, doc_fontsize_pt;                   # Adjust axis labels
xlabel          = L"$\phi$ [-]",
ylabel          = L"$f/(UD)^2$ [-]"
)
cr              = (1,5) #extrema(Ds)                # linear range for the mapping

CairoMakie.xlims!(ax, 0, 1)                         # Axis limits
CairoMakie.ylims!(ax, -1.6, 1.4)

l1 = CairoMakie.lines!(ax,
    base[:,2] ./ kin.T1 .- 1, base[:,1] ./ 64^2;
    linewidth   = 1.2,
    color       = 1, colormap = cmap, colorrange = cr,
    linestyle   = linestyles[mod1(1, length(linestyles))],
)

l2 = CairoMakie.lines!(ax,
    twice[:,2] ./ kin.T1 .- 1, twice[:,1] ./ 64^2;
    linewidth   = 1.2,
    color       = 5, colormap = cmap, colorrange = cr,
    linestyle   = linestyles[mod1(2, length(linestyles))],
)

handles = [l1, l2]
labels = ["(3D,D,D)", "(6D,D,D)"]

Legend(
    fig[1, 2],
    handles,
    labels;
    framevisible = false,
    tellheight   = false,
    labelsize    = doc_fontsize_pt,
    labelfont    = "TeX Gyre Pagella"
)                                    # Put legend in a new row below the axis

# rowgap!(fig.layout, 2)                              # Adjust the gap between the legend and the figure.
# rowsize!(fig.layout, 1, Relative(1))                # plot takes most space
# rowsize!(fig.layout, 2, Relative(0.08))             # legend ~8% of height

save(savepath, fig; pt_per_unit=1)

## TIME STEP CONVERGENCE ERROR PLOT
savepath        = "figures/convergence/dt_unfiltered_force_conv_plot.pdf"
fig             = Figure(size=(figwidth_pt, figheight_pt))
ax              = jelly_axis(fig, doc_fontsize_pt;                   # Adjust axis labels
xlabel          = L"$dt$ [-]",
ylabel          = L"rmse($f$) [-]"
)
cr              = (1,5) #extrema(Ds)                # linear range for the mapping

CairoMakie.lines!(ax,
    dts, dt_rmses;
    linewidth   = 1.2,
    color       = 1,
    colormap    = cmap,
    colorrange  = cr,
    linestyle   = linestyles[mod1(1, length(linestyles))]
)

CairoMakie.scatter!(ax,
    dts, dt_rmses;
    marker      = :circle,
    markersize  = 8,
    color       = :transparent,
    strokecolor = cgrad(cmap)[1],     # ← same exact colour
    strokewidth = 1.2
)

save(savepath, fig; pt_per_unit=1)

## Full signal plot of dt convergence
savepath        = "figures/convergence/dt_full_unfiltered_force_conv_plot.pdf"
fig             = Figure(size=(figwidth_pt, figheight_pt))
ax              = jelly_axis(fig, doc_fontsize_pt;                   # Adjust axis labels
xlabel          = L"$\phi$ [-]",
ylabel          = L"$f/(UD)^2$ [-]"
)
cr              = extrema(dts)                 # linear range for the mapping

CairoMakie.xlims!(ax, 0, 1)
# CairoMakie.ylims!(ax, -3.5, 2.5)

handles = Makie.AbstractPlot[]   # store line objects
labels  = String[]          # store legend labels

for (i, sim) in enumerate(dts_sims)
    ln = CairoMakie.lines!(
        ax,
        sim.time ./ kin.T1 .- 10,
        sim.funfiltered ./ 32^2;
        linewidth=1.2,
        color=dts[i],                 # <-- numeric value (gets colour-mapped)
        colormap=cmap,
        colorrange=cr,
        colorscale=identity,          # <-- linear (default, but explicit)
        linestyle=linestyles[mod1(i, length(linestyles))],
    )

    push!(handles, ln)
    push!(labels, "dt = $(dts[i])")
end

Legend(
    fig[1, 2],
    handles,
    labels;
    framevisible = false,
    tellheight   = false,
    labelsize    = doc_fontsize_pt,
    labelfont    = "TeX Gyre Pagella"
)

save(savepath, fig; pt_per_unit=1)