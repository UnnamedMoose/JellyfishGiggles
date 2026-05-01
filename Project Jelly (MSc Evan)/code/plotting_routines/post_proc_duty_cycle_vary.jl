## ADD VIZ AND POSTPROCESSING FUNCTIONS
include(joinpath(@__DIR__, "..", "visualisation_algorithms.jl"))

## UPLOAD THE ACCORDING FILES
files               = [
    "3D_D64_eps2_dt005_basecase_gam020.csv",
    "3D_D64_eps2_dt005_basecase_gam025.csv",
    "3D_D64_eps2_dt005_basecase_gam030.csv",
    "3D_D64_eps2_dt005_basecase_gam035.csv",
    "3D_D64_eps2_dt005_basecase_gam040.csv",
    "3D_D64_eps2_dt005_basecase_gam045.csv",
    "3D_D64_eps2_dt005_basecase_gam050.csv",
    "3D_D64_eps2_dt005_basecase_gam055.csv",
    "3D_D64_eps2_dt005_basecase_gam060.csv",
    "3D_D64_eps2_dt005_basecase_gam065.csv",
    "3D_D64_eps2_dt005_basecase_gam070.csv",
    "3D_D64_eps2_dt005_basecase_gam075.csv",
]

cases = [
    (; γ=0.2, T1=kin.T1, T2=kin.T1, Tg=0.00*kin.T1),
    (; γ=0.25, T1=kin.T1, T2=kin.T1, Tg=0.00*kin.T1),
    (; γ=0.3, T1=kin.T1, T2=kin.T1, Tg=0.00*kin.T1),
    (; γ=0.35, T1=kin.T1, T2=kin.T1, Tg=0.00*kin.T1),
    (; γ=0.4, T1=kin.T1, T2=kin.T1, Tg=0.00*kin.T1),
    (; γ=0.45, T1=kin.T1, T2=kin.T1, Tg=0.00*kin.T1),
    (; γ=0.5, T1=kin.T1, T2=kin.T1, Tg=0.00*kin.T1),
    (; γ=0.55, T1=kin.T1, T2=kin.T1, Tg=0.00*kin.T1),
    (; γ=0.60, T1=kin.T1, T2=kin.T1, Tg=0.00*kin.T1),
    (; γ=0.65, T1=kin.T1, T2=kin.T1, Tg=0.00*kin.T1),
    (; γ=0.70, T1=kin.T1, T2=kin.T1, Tg=0.00*kin.T1),
    (; γ=0.75, T1=kin.T1, T2=kin.T1, Tg=0.00*kin.T1)
]

sims            = load_simulation.("data/duty_cycle_variation_data/" .* files)
results         = run_case.(sims, cases)

## PLOTTING VELOCITIES FOR VARYING DUTY CYCLE
savepath = "figures/results/base_cases_velocities_plot.pdf"
fig      = Figure(size=(figwidth_pt, figheight_pt))
ax       = jelly_axis(fig, doc_fontsize_pt;
    xlabel = L"$\gamma$ [-]",
    ylabel = L"$u/U$ [-]"
)

x = getproperty.(cases, :γ)
y1 = -getproperty.(results, :v_mean)
y2 = -getproperty.(results, :v_peak)

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
savepath        = "figures/results/base_cases_COT_plot.pdf"
fig             = Figure(size=(figwidth_pt, figheight_pt))
ax              = jelly_axis(fig, doc_fontsize_pt;                   # Adjust axis labels
xlabel          = L"$\gamma$ [-]",
ylabel          = L"$COT/(UD)^2$ [-]"   
)

l1 = CairoMakie.lines!(ax,
    getproperty.(cases, :γ), getproperty.(results, :COT);
    linewidth   = 1.2,
    color       = 1,
    colormap    = cmap,
    colorrange  = cr,
    linestyle   = linestyles[mod1(1, length(linestyles))],
)
CairoMakie.scatter!(ax,
    getproperty.(cases, :γ), getproperty.(results, :COT);
    marker      = :circle,
    markersize  = 8,
    color       = :transparent,
    strokecolor = cgrad(cmap)[1],     # ← same exact colour
    strokewidth = 1.2,
)

save(savepath, fig; pt_per_unit=1)


df1 = CSV.read("Data/Simulation_Data/base_cases/results_3D_basecaseforces.csv", DataFrame)
df1 = CSV.read("data/duty_cycle_variation_data/3D_D64_eps2_dt005_basecase_gam040.csv", DataFrame)

# α               =   1 - exp(-0.05 / (0.03*kin.T1*64))

# Fp = zeros(length(df1[:,1]))
# Fp[1] = 0.0
# for i in 2:length(df1[:,1])-1
#     Fp[i] = (1-α)*Fp[i-1] + α*df1[:,1][i]
# end

# Plots.plot(df1[:,5], (Fp) ./ 64^2, xlims=(10,15), ylims=(-1.5,1.5))
# Plots.plot!(df1[:,5], df1[:,2] ./ 64^2)
# Plots.plot!(df1[:,5], df1[:,3] ./ 64^2)

s_per_p1            = findfirst(>(kin.T1), df1[:,2]);
idx = find_stationary_index(df1[:,4]; window_size=s_per_p1, tol_mean=1e-1, tol_std=1e-1, n_consecutive=4);
x = df1[idx:end, 2]

per_idx_local = findfirst(t -> isapprox(t, round(t); atol=1e-8), x)

per_idx = isnothing(per_idx_local) ? nothing : per_idx_local + idx - 1

## PLOTTING VELOCITIES FOR VARYING DUTY CYCLE
savepath        = "figures/results/base_case_pos_plot.pdf"
figwidth_pt     = 452.97                            # Latex \textwidth in pt from the log
aspect          = 0.62                              # Ratio of height over width
figheight_pt    = figwidth_pt * aspect              # Figure height
doc_fontsize_pt = 12.5                              # Latex font size

set_theme!(Theme(font = "TeX Gyre Pagella",   fontsize = doc_fontsize_pt,))
fig             = Figure(size=(figwidth_pt, figheight_pt))
ax              = jelly_axis(fig, doc_fontsize_pt;                   # Adjust axis labels
    xlabel = L"$\phi$ [-]",
    ylabel = L"$u_x$ [$\frac{cm}{s}$]",
)

CairoMakie.xlims!(0,20)
# CairoMakie.ylims!(-50,5)

linestyles      = [:solid, :dash, :dashdot, :dot]   # Different linestyle per line
cr              = (1,5) #extrema(Ds)                # linear range for the mapping
cmap            = :redsblues                        # pick any Makie colormap symbol

l1 = CairoMakie.lines!(ax,
    df1[:,2] ./ kin.T1, df1[:,4] .* -mean(ysp);
    linewidth   = 1.2,
    color       = 1,
    colormap    = cmap,
    colorrange  = cr,
    linestyle   = linestyles[mod1(1, length(linestyles))],
    label       = L"u"
)

l2 = CairoMakie.hlines!(ax,
    mean(df1[:,4][per_idx:end].* -mean(ysp));
    linewidth   = 1.2,
    color       = 5,
    colormap    = cmap,
    colorrange  = cr,
    linestyle   = linestyles[mod1(2, length(linestyles))],
    label       = L"\bar{u}"
)

# l3 = CairoMakie.lines!(ax,
#     df_f[:,5] ./ T1 .- 15, df_f[:,3] ./ 64^2;
#     linewidth   = 1.2,
#     color       = 5,
#     colormap    = cmap,
#     colorrange  = cr,
#     linestyle   = linestyles[mod1(3, length(linestyles))],
#     label       = "Added Mass"
# )

yl = (-5, 0)  # same as your ylims

CairoMakie.band!(ax,
    [df1[:,2][per_idx], 40],
    [yl[1], yl[1]],
    [yl[2], yl[2]];
    color = (:lightblue, 0.5),   # 25% transparency
    label = "Steady-State"
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