## ADD VIZ AND POSTPROCESSING FUNCTIONS
include(joinpath(@__DIR__, "..", "visualisation_algorithms.jl"))
## UPLOAD THE ACCORDING FILES
files               = [
    "gam040_Tg_000_new.csv",
    "gam040_Tg_025.csv",
    "gam040_Tg_050.csv",
    "gam040_Tg_075.csv",
    "gam040_Tg_100.csv",
    # "gam040_Tg_125new.csv",
    # "gam040_Tg_150.csv",
    # "gam040_Tg_200.csv",
    # "gam040_Tg_250.csv",
    # "gam040_Tg_300.csv",
    # "gam040_Tg_400.csv",
    # "gam040_Tg_500.csv",
    "gam045_Tg_000.csv",
    "gam045_Tg_025.csv",
    "gam045_Tg_050.csv",
    "gam045_Tg_075.csv",
    "gam045_Tg_100.csv",
    "gam050_Tg_000.csv",
    "gam050_Tg_025.csv",
    "gam050_Tg_050.csv",
    "gam050_Tg_075.csv",
    "gam050_Tg_100.csv"
]

cases = [
    (; γ=0.40, T1=T1, T2=1, Tg=0.00*T1),
    (; γ=0.40, T1=T1, T2=1, Tg=0.25*T1),
    (; γ=0.40, T1=T1, T2=1, Tg=0.50*T1),
    (; γ=0.40, T1=T1, T2=1, Tg=0.75*T1),
    (; γ=0.40, T1=T1, T2=1, Tg=1.00*T1),
    # (; γ=0.40, T1=T1, T2=1, Tg=1.25*T1),
    # (; γ=0.40, T1=T1, T2=1, Tg=1.50*T1),
    # (; γ=0.40, T1=T1, T2=1, Tg=2.00*T1),
    # (; γ=0.40, T1=T1, T2=1, Tg=2.50*T1),
    # (; γ=0.40, T1=T1, T2=1, Tg=3.00*T1),
    # (; γ=0.40, T1=T1, T2=1, Tg=4.00*T1),
    # (; γ=0.40, T1=T1, T2=1, Tg=5.00*T1),
    (; γ=0.45, T1=T1, T2=1, Tg=0.00*T1),
    (; γ=0.45, T1=T1, T2=1, Tg=0.25*T1),
    (; γ=0.45, T1=T1, T2=1, Tg=0.50*T1),
    (; γ=0.45, T1=T1, T2=1, Tg=0.75*T1),
    (; γ=0.45, T1=T1, T2=1, Tg=1.00*T1),
    (; γ=0.50, T1=T1, T2=1, Tg=0.00*T1),
    (; γ=0.50, T1=T1, T2=1, Tg=0.25*T1),
    (; γ=0.50, T1=T1, T2=1, Tg=0.50*T1),
    (; γ=0.50, T1=T1, T2=1, Tg=0.75*T1),
    (; γ=0.50, T1=T1, T2=1, Tg=1.00*T1),
]

sims            = load_simulation.("data/varying_Tg/" .* files)
results         = run_case.(sims, cases)

## PLOTTING VELOCITIES FOR VARYING GLIDING TIME
savepath        = "figures/results/varyTg_velocities_plot.pdf"
figwidth_pt     = 452.97                            # Latex \textwidth in pt from the log
aspect          = 0.62                              # Ratio of height over width
figheight_pt    = figwidth_pt * aspect              # Figure height
doc_fontsize_pt = 12.5                              # Latex font size

set_theme!(Theme(font = "TeX Gyre Pagella",   fontsize = doc_fontsize_pt,))
fig             = Figure(size=(figwidth_pt, figheight_pt))
ax              = jelly_axis(fig, doc_fontsize_pt;                   # Adjust axis labels
xlabel          = L"$T_g$ [s]",
ylabel          = L"$u/U$ [-]"
)

x = getproperty.(cases, :Tg)[1:7] ./ kin.T1
y1 = -getproperty.(results, :v_mean)[1:7]
y2 = -getproperty.(results, :v_peak)[1:7]

c1 = cgrad(cmap)[1]
c2 = cgrad(cmap)[3]

CairoMakie.lines!(ax, x, y1;
    linewidth  = 1.2,
    color      = c1,
    linestyle  = linestyles[mod1(1, length(linestyles))]
)
CairoMakie.scatter!(ax, x, y1;
    marker      = :circle,
    markersize  = 8,
    color       = :transparent,
    strokecolor = c1,
    strokewidth = 1.2,
)

CairoMakie.lines!(ax, x, y2;
    linewidth  = 1.2,
    color      = c2,
    linestyle  = linestyles[mod1(2, length(linestyles))]
)
CairoMakie.scatter!(ax, x, y2;
    marker      = :utriangle,   # or :rect
    markersize  = 8,
    color       = :transparent,
    strokecolor = c2,
    strokewidth = 1.2,
)

legend_elements = [
    [LineElement(color=c1, linestyle=linestyles[mod1(1, length(linestyles))], linewidth=1.2),
     MarkerElement(marker=:circle, markersize=8, color=:transparent, strokecolor=c1, strokewidth=1.2)],

    [LineElement(color=c2, linestyle=linestyles[mod1(2, length(linestyles))], linewidth=1.2),
     MarkerElement(marker=:utriangle, markersize=8, color=:transparent, strokecolor=c2, strokewidth=1.2)]
]

legend_labels = [L"$\bar{u}$", L"$u_{\mathrm{peak}}$"]

leg = Legend(fig, legend_elements, legend_labels;
    orientation = :horizontal,
    framevisible = false,
    labelsize = doc_fontsize_pt
)

fig[2, 1] = leg

rowgap!(fig.layout, 2)
rowsize!(fig.layout, 1, Relative(1))
rowsize!(fig.layout, 2, Relative(0.08))

save(savepath, fig; pt_per_unit=1)

## PLOTTING COT FOR VARYING DUTY CYCLE
savepath        = "figures/results/varyTg_COT_plot.pdf"
fig             = Figure(size=(figwidth_pt, figheight_pt))
ax              = jelly_axis(fig, doc_fontsize_pt;                   # Adjust axis labels
xlabel          = L"$T_g$ [-]",
ylabel          = L"$COT/(UD)^2$ [-]"
)

l1 = CairoMakie.lines!(ax,
    getproperty.(cases, :Tg) ./ T1, getproperty.(results, :COT);
    linewidth   = 1.2,
    color       = 1,
    colormap    = cmap,
    colorrange  = cr,
    linestyle   = linestyles[mod1(1, length(linestyles))],
)
CairoMakie.scatter!(ax,
    getproperty.(cases, :Tg) ./ T1, getproperty.(results, :COT);
    marker      = :circle,
    markersize  = 8,
    color       = :transparent,
    strokecolor = cgrad(cmap)[1],     # ← same exact colour
    strokewidth = 1.2,
)

save(savepath, fig; pt_per_unit=1)

## PLOTTING COT FOR VARIOUS SINGALS FOR MORE GLIDING TIMES
savepath        = "figures/results/varyTg_full_acc_signal.pdf"
fig             = Figure(size=(figwidth_pt, figheight_pt))
ax              = jelly_axis(fig, doc_fontsize_pt;                   # Adjust axis labels
xlabel          = L"$T_g$ [-]",
ylabel          = L"$u/U$ [-]"
)
Tgs             = [0,1.5]
cr              = extrema(Tgs)                # linear range for the mapping
handles         = Makie.AbstractPlot[]   # store line objects
labels          = String[]          # store legend labels

CairoMakie.xlims!(ax, 15, 20)

for (i, sim) in enumerate(sims)
    ln = CairoMakie.lines!(
        ax,
        sim.time ./ T1,
        sim.acc;
        linewidth=1.2,
        color=Tgs[i],                 # <-- numeric value (gets colour-mapped)
        colormap=cmap,
        colorrange=cr,
        colorscale=identity,          # <-- linear (default, but explicit)
        linestyle=linestyles[mod1(i, length(linestyles))],
        # label=LaTeXString("\\gamma = $(cfg.γ)")
    )

    push!(handles, ln)
    push!(labels, "Tg = $(Tgs[i])")
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

## Three duty cycle lines:
savepath        = "figures/results/varygamandTg_COT_plot.pdf"
fig             = Figure(size=(figwidth_pt, figheight_pt))
ax              = jelly_axis(fig, doc_fontsize_pt;                   # Adjust axis labels
    xlabel = L"$T_g$ [s]",
    ylabel = L"$COT/(UD)^2$ [-]"
)
# cr              = (1,5) #extrema(Ds)                # linear range for the mapping
γ_groups        = (1:5, 6:10, 11:15)
cols = cgrad(cmap, 5, categorical=true)
cols = [cols[1], RGB(0.45, 0.05, 0.45), cols[5]]
markers = [:circle, :utriangle, :rect]

for (k, idxs) in enumerate(γ_groups)
    Tg  = getproperty.(cases[idxs],   :Tg)  ./ kin.T1
    cot = getproperty.(results[idxs], :COT)

    CairoMakie.lines!(ax, Tg, cot;
        linewidth = 1.2,
        color     = cols[k],
        linestyle = linestyles[mod1(k, length(linestyles))],
        # label     = L"\gamma = %$(getproperty(cases[first(idxs)], :γ))",
    )
    CairoMakie.scatter!(ax, Tg, cot;
        marker      = :circle,
        markersize  = 8,
        color       = :transparent,
        strokecolor = cols[k],
        strokewidth = 1.2,
    )
end


legend_elements = [
    [LineElement(color=cols[1], linestyle=linestyles[mod1(1, length(linestyles))], linewidth=1.2),
     MarkerElement(marker=:circle, markersize=8, color=:transparent, strokecolor=cols[1], strokewidth=1.2)],

    [LineElement(color=cols[2], linestyle=linestyles[mod1(2, length(linestyles))], linewidth=1.2),
     MarkerElement(marker=:utriangle, markersize=8, color=:transparent, strokecolor=cols[2], strokewidth=1.2)],

    [LineElement(color=cols[3], linestyle=linestyles[mod1(3, length(linestyles))], linewidth=1.2),
     MarkerElement(marker=:rect, markersize=8, color=:transparent, strokecolor=cols[3], strokewidth=1.2)]
]

legend_labels = [L"\gamma = 0.40", L"\gamma = 0.45", L"\gamma = 0.50"]

leg = Legend(fig, legend_elements, legend_labels;
    orientation = :horizontal,
    framevisible = false,
    labelsize = doc_fontsize_pt
)

fig[2, 1] = leg                                     # Put legend in a new row below the axis

rowgap!(fig.layout, 2)                              # Adjust the gap between the legend and the figure.
rowsize!(fig.layout, 1, Relative(1))                # plot takes most space
rowsize!(fig.layout, 2, Relative(0.08))             # legend ~8% of height

save(savepath, fig; pt_per_unit=1)