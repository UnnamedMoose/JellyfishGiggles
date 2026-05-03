## ADD VIZ AND POSTPROCESSING FUNCTIONS
include(joinpath(@__DIR__, "..", "visualisation_algorithms.jl"))

## UPLOAD THE ACCORDING FILES
files               = [
    "gam040_T1_1_T2_050_new.csv",   # correct
    "gam040_T1_1_T2_075.csv",       # correct
    "gam040_T1_1_T2_100_new.csv",   # correct
    "gam040_T1_1_T2_125.csv",       # correct
    "gam040_T1_1_T2_150.csv",       # correct
    "gam040_T1_1_T2_175.csv",       # correct
    "gam040_T1_1_T2_200.csv",       # correct
    "gam045_T1_1_T2_050.csv",
    "gam045_T1_1_T2_075.csv",
    "gam045_T1_1_T2_100.csv",
    "gam045_T1_1_T2_125.csv",
    "gam045_T1_1_T2_150.csv",
    "gam045_T1_1_T2_175.csv",
    "gam045_T1_1_T2_200.csv",
    "gam050_T1_1_T2_050.csv",
    "gam050_T1_1_T2_075.csv",
    "gam050_T1_1_T2_100.csv",
    "gam050_T1_1_T2_125.csv",
    "gam050_T1_1_T2_150.csv",
    "gam050_T1_1_T2_175.csv",
    "gam050_T1_1_T2_200.csv",
]
T1 = kin.T1
cases = [
    (; γ=0.4, T1=T1, T2=0.5*T1, Tg=0),
    (; γ=0.4, T1=T1, T2=0.75*T1, Tg=0),
    (; γ=0.4, T1=T1, T2=1.00*T1, Tg=0),
    (; γ=0.4, T1=T1, T2=1.25*T1, Tg=0),
    (; γ=0.4, T1=T1, T2=1.5*T1, Tg=0),
    (; γ=0.4, T1=T1, T2=1.75*T1, Tg=0),
    (; γ=0.4, T1=T1, T2=2*T1, Tg=0),
    (; γ=0.45, T1=T1, T2=0.5*T1, Tg=0),
    (; γ=0.45, T1=T1, T2=0.75*T1, Tg=0),
    (; γ=0.45, T1=T1, T2=1.00*T1, Tg=0),
    (; γ=0.45, T1=T1, T2=1.25*T1, Tg=0),
    (; γ=0.45, T1=T1, T2=1.5*T1, Tg=0),
    (; γ=0.45, T1=T1, T2=1.75*T1, Tg=0),
    (; γ=0.45, T1=T1, T2=2*T1, Tg=0),
    (; γ=0.5, T1=T1, T2=0.5*T1, Tg=0),
    (; γ=0.5, T1=T1, T2=0.75*T1, Tg=0),
    (; γ=0.5, T1=T1, T2=1.00*T1, Tg=0),
    (; γ=0.5, T1=T1, T2=1.25*T1, Tg=0),
    (; γ=0.5, T1=T1, T2=1.5*T1, Tg=0),
    (; γ=0.5, T1=T1, T2=1.75*T1, Tg=0),
    (; γ=0.5, T1=T1, T2=2*T1, Tg=0),
]

sims                = load_simulation.("data/varying_T1T2/" .* files)
results             = run_case.(sims, cases)

## PLOT VELOCITIES FOR VARYING PERIODS
savepath        = "figures/results/varyT1T2_velocities_plot.pdf"
fig             = Figure(size=(figwidth_pt, figheight_pt))
ax              = jelly_axis(fig, doc_fontsize_pt;                   # Adjust axis labels
xlabel          = L"$T_2$ [s]",
ylabel          = L"$u/U$ [-]"
)

x = getproperty.(cases, :T2)[1:7] ./ kin.T1
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
savepath        = "figures/results/varyT1T2_COT_plot.pdf"
fig             = Figure(size=(figwidth_pt, figheight_pt))
ax              = jelly_axis(fig, doc_fontsize_pt;                   # Adjust axis labels
xlabel          = L"$T_2$ [s]",
ylabel          = L"$COT/(UD)^2$ [-]"
)

l1 = CairoMakie.lines!(ax,
    getproperty.(cases, :T2) ./ T1, getproperty.(results, :COT);
    linewidth   = 1.2,
    color       = 1,
    colormap    = cmap,
    colorrange  = cr,
    linestyle   = linestyles[mod1(1, length(linestyles))],
)
CairoMakie.scatter!(ax,
    getproperty.(cases, :T2) ./ T1, getproperty.(results, :COT);
    marker      = :circle,
    markersize  = 8,
    color       = :transparent,
    strokecolor = cgrad(cmap)[1],     # ← same exact colour
    strokewidth = 1.2,
)

save(savepath, fig; pt_per_unit=1)

## Three duty cycle lines:
savepath        = "figures/results/varygamandT2_COT_plot.pdf"
fig             = Figure(size=(figwidth_pt, figheight_pt))
ax              = jelly_axis(fig, doc_fontsize_pt;                   # Adjust axis labels
xlabel          = L"$T_2$ [s]",
ylabel          = L"$COT/(UD)^2$ [-]"
)

γ_groups = (1:7, 8:14, 15:21)

cols = cgrad(cmap, 5, categorical=true)
cols = [cols[1], RGB(0.45, 0.05, 0.45), cols[5]]
markers = [:circle, :utriangle, :rect]

for (k, idxs) in enumerate(γ_groups)
    T2  = getproperty.(cases[idxs],   :T2)  ./ kin.T1
    cot = getproperty.(results[idxs], :COT)

    CairoMakie.lines!(ax, T2, cot;
        linewidth = 1.2,
        color     = cols[k],
        linestyle = linestyles[mod1(k, length(linestyles))],
        # label     = L"\gamma = %$(getproperty(cases[first(idxs)], :γ))",
    )
    CairoMakie.scatter!(ax, T2, cot;
        marker      = markers[k],
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