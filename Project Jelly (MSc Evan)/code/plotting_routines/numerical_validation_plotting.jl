include(joinpath(@__DIR__, "..", "visualisation_algorithms.jl"))
"""
--- NUMERICAL VALIDATION PLOTTING ROUTINES ---
Load the appropriate files
"""
## Load data
BiotBCs_3D              = CSV.read("data/boundary_condition_data/biot_savart_bc_check.csv", DataFrame)
SymBCs_3D               = CSV.read("data/boundary_condition_data/symmetrical_bc_check.csv", DataFrame)

BiotBCs_fulljelly_2D    = CSV.read("data/boundary_condition_data/2D_full_jelly_biot_savart_bc.csv", DataFrame)
BiotBCs_halfjelly_2D    = CSV.read("data/boundary_condition_data/2D_half_jelly_biot_bc.csv", DataFrame)
symBCs_halfjelly_2D     = CSV.read("data/boundary_condition_data/symmetrical_bc_check.csv", DataFrame)

filtering               = CSV.read("data/boundary_condition_data/firstcolumn_filteredforce_finalcolumn_actualforce.csv", DataFrame)

"""
Plotting Routines
"""

## PLOT BOUNDARY CONDITION CHECKS
savepath        = "figures/methodology/symmetricalBC_check_2D.pdf"
fig             = Figure(size=(figwidth_pt, figheight_pt))
ax              = jelly_axis(fig, doc_fontsize_pt;                   # Adjust axis labels
xlabel          = L"$\phi$ [-]",
ylabel          = L"$f/(DU^2)$ [-]"
)

CairoMakie.xlims!(ax, 0, 5)                         # Axis limits
# CairoMakie.ylims!(ax, -3.5, 2.5)

# l1 = CairoMakie.lines!(ax,
#     BiotBCs_3D[:,2] ./ kin.T1, BiotBCs_3D[:,1] ./ 64^2;
#     linewidth   = 1.2,
#     color       = 1, colormap = cmap, colorrange = cr,
#     linestyle   = linestyles[mod1(1, length(linestyles))],
#     label       = "BiotSavart BCs"
# )

l2 = CairoMakie.lines!(ax,
    BiotBCs_fulljelly_2D[:,2] ./ kin.T1, BiotBCs_fulljelly_2D[:,4] ./ 32 .* 2*shape_area(cps);
    linewidth   = 1.2,
    color       = 5, colormap = cmap, colorrange = cr,
    linestyle   = linestyles[mod1(2, length(linestyles))],
    label       = "Biot-Savart"
)

l3 = CairoMakie.lines!(ax,
    symBCs_halfjelly_2D[:,2] ./ kin.T1, symBCs_halfjelly_2D[:,4] ./ 32 .* 2*shape_area(cps);
    linewidth   = 1.2,
    color       = 1, colormap = cmap, colorrange = cr,
    linestyle   = linestyles[mod1(2, length(linestyles))],
    label       = "Symmetrical"
)

# axislegend(ax; position = :rt, framevisible = false, labelsize = doc_fontsize_pt)             # Legend in figure itself.

leg = Legend(fig, ax;
    orientation  = :horizontal,
    # nbanks       = 2,                               # Number of rows
    framevisible =  false,                          # Turn on/off of frame around legend
    labelsize    = doc_fontsize_pt                  # Label size the same as document Font
)

fig[2, 1] = leg                                     # Put legend in a new row below the axis

rowgap!(fig.layout, 2)                              # Adjust the gap between the legend and the figure.
rowsize!(fig.layout, 1, Relative(1))                # plot takes most space
rowsize!(fig.layout, 2, Relative(0.08))             # legend ~8% of height

save(savepath, fig; pt_per_unit=1)

idx1 = 922
idxend = 865
filtering[:,2][922] - filtering[:,2][865]

s_per_p1            = findfirst(>(0), filtering[:,end][200:end])
idx = find_stationary_index(df1[:,4]; window_size=s_per_p1, tol_mean=1e-1, tol_std=1e-1, n_consecutive=4);
x = df1[idx:end, 2]

per_idx_local = findfirst(t -> isapprox(t, round(t); atol=1e-8), x)

per_idx = isnothing(per_idx_local) ? nothing : per_idx_local + idx - 1


## PLOT FILTERED AND NON-FILTERED FORCE
savepath        = "figures/methodology/force_filtering.pdf"
fig             = Figure(size=(figwidth_pt, figheight_pt))
ax              = jelly_axis(fig, doc_fontsize_pt;                   # Adjust axis labels
xlabel          = L"$\phi$ [-]",
ylabel          = L"$f/(UD)^2$ [-]"
)

CairoMakie.xlims!(ax, 0, 2)                         # Axis limits
CairoMakie.ylims!(ax, -2, 2)

l1 = CairoMakie.lines!(ax,
    filtering[:,2] ./ kin.T1, filtering[:,1] ./64^2;
    linewidth   = 1.2,
    color       = 1, colormap = cmap, colorrange = cr,
    linestyle   = linestyles[mod1(1, length(linestyles))],
    label       = "filtered"
)

l2 = CairoMakie.lines!(ax,
    filtering[:,2] ./ kin.T1, filtering[:,end] ./64^2;
    linewidth   = 1.2,
    color       = 5, colormap = cmap, colorrange = cr,
    linestyle   = linestyles[mod1(2, length(linestyles))],
    label       = "non-filtered"
)

yl = (-2, 2)  # same as your ylims

CairoMakie.band!(ax,
    [filtering[:,2][865] ./ kin.T1, filtering[:,2][922] ./ kin.T1],
    [yl[1], yl[1]],
    [yl[2], yl[2]];
    color = (:lightblue, 0.5),   # 25% transparency
    label = L"$\Delta \phi$"
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

df1 = CSV.read("data/simulation_data/3D_full_with_newforce.csv", DataFrame)
df2 = CSV.read("data/duty_cycle_variation_data/3D_D64_eps2_dt005_basecase_gam040.csv", DataFrame)
df3 = CSV.read("data/boundary_condition_data/results_2D.csv", DataFrame)
df4 = CSV.read("data/simulation_data/full_jellyfish_geometry.csv", DataFrame )
α = 0.15
Fp = zeros(length(df3[:,1]))
Fp[1] = 0.0
for i in 2:length(df3[:,1])-1
    Fp[i] = (1-α)*Fp[i-1] + α*df3[:,1][i]
end
Plots.plot(df2[:,2] ./ kin.T1, df2[:,1] ./ shape_volume(cps), xlims=(0,5))
Plots.plot!(df3[:,2] ./ kin.T1, cumsum(0.05* 2 .*Fp ./ shape_area(cps)))

Plots.plot(df2[:,2] ./ kin.T1, df2[:,4])
Plots.plot!(df3[:,2] ./ kin.T1, df3[:,3])


savepath        = "figures/results/2Dover3Dcomparison_new.pdf"
fig             = Figure(size=(figwidth_pt, figheight_pt))
ax              = jelly_axis(fig, doc_fontsize_pt;                   # Adjust axis labels
xlabel          = L"$\phi$ [-]",
ylabel          = L"$f/(D^n U^2)$ [-]"
)

CairoMakie.xlims!(ax, 0, 3)                              # Axis limits
# CairoMakie.ylims!(ax, -3.5, 2.5)

# l1 = CairoMakie.lines!(ax,
#     BiotBCs_3D[:,2] ./ kin.T1, BiotBCs_3D[:,1] ./ 64^2;
#     linewidth   = 1.2,
#     color       = 1, colormap = cmap, colorrange = cr,
#     linestyle   = linestyles[mod1(1, length(linestyles))],
#     label       = "BiotSavart BCs"
# )

l2 = CairoMakie.lines!(ax,
    df2[:,2] ./ kin.T1 .- 5, df2[:,1] ./ 64^2; #shape_volume(cps);
    linewidth   = 1.2,
    color       = 1, colormap = cmap, colorrange = cr,
    linestyle   = linestyles[mod1(1, length(linestyles))],
    label       = "3D Quarter"
)

l1 = CairoMakie.lines!(ax,
    df1[:,2] ./ kin.T1 .- 5, df1[:,1] ./ 64^2;
    linewidth   = 1.2,
    color       = 2, colormap = cmap, colorrange = cr,
    linestyle   = linestyles[mod1(2, length(linestyles))],
    label       = "3D Full"
)

l3 = CairoMakie.lines!(ax,
    df3[:,2] ./ kin.T1 .- 5, (2 .*Fp) ./ 64; #shape_area(cps);
    linewidth   = 1.2,
    color       = 5, colormap = cmap, colorrange = cr,
    linestyle   = linestyles[mod1(3, length(linestyles))],
    label       = "2D"
)

# axislegend(ax; position = :rt, framevisible = false, labelsize = doc_fontsize_pt)             # Legend in figure itself.

leg = Legend(fig, ax;
    orientation  = :horizontal,
    # nbanks       = 2,                               # Number of rows
    framevisible =  false,                          # Turn on/off of frame around legend
    labelsize    = doc_fontsize_pt                  # Label size the same as document Font
)

fig[2, 1] = leg                                     # Put legend in a new row below the axis

rowgap!(fig.layout, 2)                              # Adjust the gap between the legend and the figure.
rowsize!(fig.layout, 1, Relative(1))                # plot takes most space
rowsize!(fig.layout, 2, Relative(0.08))             # legend ~8% of height

save(savepath, fig; pt_per_unit=1)


savepath        = "figures/results/vel_2D3D_comparison_new.pdf"
fig             = Figure(size=(figwidth_pt, figheight_pt))
ax              = jelly_axis(fig, doc_fontsize_pt;                   # Adjust axis labels
xlabel          = L"$\phi$ [-]",
ylabel          = L"$u/U$ [-]"
)

CairoMakie.xlims!(ax, 0, 3)                         # Axis limits
CairoMakie.ylims!(ax, -2.5, 0)

l2 = CairoMakie.lines!(ax,
    df2[:,2] ./ kin.T1 .- 5, df2[:,4]; #shape_volume(cps);
    linewidth   = 1.2,
    color       = 1, colormap = cmap, colorrange = cr,
    linestyle   = linestyles[mod1(1, length(linestyles))],
    label       = "3D Quarter"
)

l1 = CairoMakie.lines!(ax,
    df1[:,2] ./ kin.T1 .- 5, df1[:,4];
    linewidth   = 1.2,
    color       = 2, colormap = cmap, colorrange = cr,
    linestyle   = linestyles[mod1(2, length(linestyles))],
    label       = "3D Full"
)

l3 = CairoMakie.lines!(ax,
    df3[:,2] ./ kin.T1 .- 4.95, cumsum(0.05* 2 .*Fp ./ shape_area(cps)); #shape_area(cps);
    linewidth   = 1.2,
    color       = 5, colormap = cmap, colorrange = cr,
    linestyle   = linestyles[mod1(2, length(linestyles))],
    label       = "2D"
)

# axislegend(ax; position = :rt, framevisible = false, labelsize = doc_fontsize_pt)             # Legend in figure itself.

leg = Legend(fig, ax;
    orientation  = :horizontal,
    # nbanks       = 2,                               # Number of rows
    framevisible =  false,                          # Turn on/off of frame around legend
    labelsize    = doc_fontsize_pt                  # Label size the same as document Font
)

fig[2, 1] = leg                                     # Put legend in a new row below the axis

rowgap!(fig.layout, 2)                              # Adjust the gap between the legend and the figure.
rowsize!(fig.layout, 1, Relative(1))                # plot takes most space
rowsize!(fig.layout, 2, Relative(0.08))             # legend ~8% of height

save(savepath, fig; pt_per_unit=1)
