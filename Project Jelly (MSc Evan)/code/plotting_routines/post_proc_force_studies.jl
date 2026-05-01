## Helper functions:
# idxn                = findfirst(isnan, signal) 
# idxp1               = findfirst(>(period), signal)
# s_per_p1            = findfirst(>(period), time1);
# idx1                = find_stationary_index(signal; window_size=s_per_p1, tol_mean=1e-2, tol_std=1e-2, n_consecutive=4);
# rms                 = sqrt(mean((force1[findfirst(>(1*period), time1):findfirst(>(4*period), time1)-1] ./ 32^2) .^2))
# m                   = mean(vel1[idx1:end]);

## Added mass check
cps0            = SMatrix{2,geom.Ncps,Float64}(cps_at_time(pathing, geom.Ncps, 0))
added_mass      = 1/2 * shape_volume(cps0); α_es = added_mass / shape_volume(cps0)
added_mass_sp   = 1/2 * 4/3 * π * (1.25/2)^3; α_sp = added_mass_sp / shape_volume(cps0)
r = maximum(cps0[2,:]); h = maximum(cps0[1,:]); α = (h / r)^1.4
added_mass_hemi = α * shape_volume(cps0)

αs = [α_es, α_sp, α]

velocities = Vector([])
for j in 1:3
    velocity = Float64[]
    vel = 0
    for i in 2:length(sims[5].force)
        # accel = (sims[5].force[i] + αs[j]*sims[5].vol[i]*sims[5].acc[i-1]) / ((1+αs[j])*sims[5].vol[i])
        accel = (sims[5].force[i]) / ((1+αs[j])*sims[5].vol[i])
        @show αs[j]*sims[5].vol[i]*sims[5].acc[i-1] / sims[5].force[i]  
        vel += accel * 0.05
        push!(velocity, vel)
    end
    push!(velocities, velocity)
end

##
df_f = CSV.read("Data/Simulation_Data/base_cases/results_3D_basecaseforces.csv", DataFrame)

α               =   1 - exp(-0.05 / (0.03*kin.T1*64))

Fp = zeros(length(df_f[:,1]))
Fp[1] = 0.0
for i in 2:length(df_f[:,1])-1
    Fp[i] = (1-α)*Fp[i-1] + α*df_f[:,1][i]
end

s_per_p1            = findfirst(>(period), df_f[:,5]);
idx = find_stationary_index(df_f[:,7]; window_size=s_per_p1, tol_mean=1e-1, tol_std=1e-1, n_consecutive=4);
x = df_f[idx:end, 5]

per_idx_local = findfirst(t -> isapprox(t, round(t); atol=1e-8), x)

per_idx = isnothing(per_idx_local) ? nothing : per_idx_local + idx - 1

## Load Data
dfnonmovgrid = CSV.read("Main/results_3D_nonmovinggrid.csv", DataFrame)

am              =   1/2 * 4/3 * π * (64/2)^3
cps0            = SMatrix{2,geom.Ncps,Float64}(cps_at_time(pathing, geom.Ncps, 0))
added_mass      = 1/2 * shape_volume(cps0 .*64);
α_es = added_mass / shape_volume(cps0.*64)
α_sp = added_mass_sp / shape_volume(cps0.*64)
r = maximum(cps0[2,:].*64); h = maximum(cps0[1,:].*64); 
α = (h / r)^1.4
added_mass_hemi = α * shape_volume(cps0.*64)

ams = [am, added_mass, added_mass_hemi] ./ shape_volume(cps0.*64)

df1 = CSV.read("Data/Simulation_Data/base_cases/3D_D64_eps2_dt005_basecase_gam040.csv", DataFrame)
df2 = CSV.read("Data/Simulation_Data/base_cases/results_3D_0.5 (1).csv", DataFrame)
df3 = CSV.read("Data/Simulation_Data/base_cases/results_3D_0.7 (1).csv", DataFrame)
df4 = CSV.read("Data/Simulation_Data/base_cases/results_3D_2.59 (1).csv", DataFrame)
dfs = [df1, df2, df3, df4]
a_am = []
for i in 2:length(df1[:,1])
    accel = (df1[:,1][i] + 2.36*shape_volume(cps0.*64) * df1[:,3][i-1]) / ((1+2.36)*shape_volume(cps0.*64))
    push!(a_am, accel)
end

Plots.plot(df1[:,2][2:end], df1[:,3][2:end])
Plots.plot!(df1[:,2][2:end], a_am)

savepath        = "Data/Figures/results/added_mass_variations.pdf"
figwidth_pt     = 452.97                            # Latex \textwidth in pt from the log
aspect          = 0.62                              # Ratio of height over width
figheight_pt    = figwidth_pt * aspect              # Figure height
doc_fontsize_pt = 12.5                              # Latex font size

set_theme!(Theme(font = "TeX Gyre Pagella",   fontsize = doc_fontsize_pt,))
fig             = Figure(size=(figwidth_pt, figheight_pt))
ax              = jelly_axis(fig, doc_fontsize_pt;                   # Adjust axis labels
    xlabel = L"$\phi$ [-]",
    ylabel = L"$\frac{u}{U}$ [-]"
)

linestyles      = [:solid, :dash, :dashdot, :dot]   # Different linestyle per line
cr              = (1,5) #extrema(Ds)                # linear range for the mapping
cmap            = :redsblues                        # pick any Makie colormap symbol
ams = [L"\alpha_{0}", L"$\alpha_{sphere,empty}$", L"$\alpha_{sphere}$", L"$\alpha_{hemisphere}$"]

CairoMakie.xlims!(ax, 0, 10)
# CairoMakie.ylims!(ax, -3.5, 2.5)

cr = (1,4)                 # linear range for the mapping
cmap = :redsblues                  # pick any Makie colormap symbol

handles = Makie.AbstractPlot[]   # store line objects
labels  = []          # store legend labels

for (i, df) in enumerate(dfs)
    ln = CairoMakie.lines!(
        ax,
        df[:,2] ./ kin.T1,
        df[:,4];
        linewidth=1.2,
        color=i,                 # <-- numeric value (gets colour-mapped)
        colormap=cmap,
        colorrange=cr,
        colorscale=identity,          # <-- linear (default, but explicit)
        linestyle=linestyles[mod1(i, length(linestyles))],
        # label=LaTeXString("\\gamma = $(cfg.γ)")
    )

    push!(handles, ln)
    push!(labels, ams[i])
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


df2 = CSV.read("Main/newforceequilibrium.csv", DataFrame)
df = CSV.read("Main/newforceeqwithTg.csv", DataFrame)

Re = 302
Cd = 24 / (Re^0.7)
Cd = (24/Re)*(1 + 0.15*Re^0.687) 
A_proj = π*maximum(cps0[:,2] .* 64)^2
Fdrag = Cd*0.5*A_proj .* df1[:,4].^2
Fam = 2.59 * shape_volume(cps0.*64) .* df1[:,3]
Ftot = df1[:,1] .+ Fdrag
a = Ftot / ((1+2.59) .* shape_volume(cps0.*64))

Plots.plot(df1[:,2], Fdrag)
Plots.plot!(df1[:,2], df1[:,1])
Plots.plot!(df1[:,2], Fam)
(force + 0.5*(24/Re^0.7)*(π*maximum(cps[:,2])^2)*v0^2) / ((1+2.59)*vol)

df3 = CSV.read("Data/Simulation_Data/varying_Tg/gam040_Tg_100.csv", DataFrame)

df = CSV.read("Main/force_comparisons.csv", DataFrame)
Plots.plot(df[:,5], df[:,1])
Plots.plot(df[:,5], df[:,2])
Plots.plot(df[:,5], df[:,3])
Plots.plot(df[:,5], df[:,4])

df = CSV.read("Main/results_3D_11.616.csv", DataFrame)
df1 = CSV.read("results_3D_Cd1.0.csv", DataFrame)
df2 = CSV.read("results_3D_Cd1.5.csv", DataFrame)
df3 = CSV.read("results_3D_Cd2.0.csv", DataFrame)
df4 = CSV.read("results_3D_Cd2.5.csv", DataFrame)
df = CSV.read("Vary_Re_Try.csv", DataFrame)

Cd = 24/(302^0.7)
D = (Cd * 0.5 * π * (64/2) ^ 2) .* (df1[:,7]).^2

nu = (64*1) / 302
Res = -(df[:,7][500:end] .* 64) ./ nu
Cd = 24 ./ (Res .^0.7)
D = (Cd * 0.5 * π * (64/2) ^ 2) .* (df[:,7][500:end]).^2

α               =   1 - exp(-0.05 / (0.03*kin.T1*64))

Fp = zeros(length(df1[:,1]))
Fp[1] = 0.0
for i in 2:length(df1[:,1])-1
    Fp[i] = (1-α)*Fp[i-1] + α*df1[:,1][i]
end

Ftot = Fp[11:end] .+ D[11:end]
cps         =   cps_at_time(pathing, geom.Ncps, 0;).* 64
acc = Ftot / (1.5*shape_volume(cps))
vel = cumsum(acc .* 0.05)


Plots.plot(df1[:,5] ./ kin.T1, df1[:,8] ./ 64, xlabel="Φ [-]", ylabel="p/D", label="Cd=1.0")
Plots.plot!(df2[:,5] ./ kin.T1, df2[:,8] ./ 64, label="Cd=1.5")
Plots.plot!(df3[:,5] ./ kin.T1, df3[:,8] ./ 64, label="Cd=2.0")
Plots.plot!(df4[:,5] ./ kin.T1, df4[:,8] ./ 64, label="Cd=2.5")

Plots.plot(df1[:,5] ./ kin.T1, df1[:,7], xlabel="Φ [-]", ylabel="u/U", label="Cd=1.0")
Plots.plot!(df2[:,5] ./ kin.T1, df2[:,7], label="Cd=1.5")
Plots.plot!(df3[:,5] ./ kin.T1, df3[:,7], label="Cd=2.0")
Plots.plot!(df4[:,5] ./ kin.T1, df4[:,7], label="Cd=2.5")


a = [df1[:,2][end], df2[:,2][end], df3[:,2][end], (24/302^0.7) * 0.5 * π*(32)^2 .* 0.25^2, (24/302^0.7) * 0.5 * π*(32)^2 .* 0.50^2, (24/302^0.7) * 0.5 * π*(32)^2 .* 0.75^2]
b = a ./ 64^2

c = []



Ftot = df[:,1][2476:end].+ df[:,4][2476:end]
cps         =   cps_at_time(pathing, geom.Ncps, 0;) .* D
acc = Ftot / (1.5*shape_volume(cps))
vel = cumsum(acc .* 0.05)

## PLOTTING VELOCITIES FOR VARYING DUTY CYCLE
savepath        = "Data/Figures/results/hybrid_approach_velocity_plot.pdf"
figwidth_pt     = 452.97                            # Latex \textwidth in pt from the log
aspect          = 0.62                              # Ratio of height over width
figheight_pt    = figwidth_pt * aspect              # Figure height
doc_fontsize_pt = 12.5                              # Latex font size

set_theme!(Theme(font = "TeX Gyre Pagella",   fontsize = doc_fontsize_pt,))
fig             = Figure(size=(figwidth_pt, figheight_pt))
ax              = jelly_axis(fig, doc_fontsize_pt;                   # Adjust axis labels
    xlabel = L"$\phi$ [-]",
    ylabel = L"$u/U$ [-]"
)
CairoMakie.xlims!(0,6)
CairoMakie.ylims!(-1.2,0)
linestyles      = [:solid, :dash, :dashdot, :dot]   # Different linestyle per line
cr              = (1,5) #extrema(Ds)                # linear range for the mapping
cmap            = :redsblues                        # pick any Makie colormap symbol

# l1 = CairoMakie.lines!(ax,
#     df[:,5] ./ kin.T1 .- 12, Fp / 64^2;
#     linewidth   = 1.2,
#     color       = 1,
#     colormap    = cmap,
#     colorrange  = cr,
#     linestyle   = linestyles[mod1(1, length(linestyles))],
#     label       = "pressure"
# )

l2 = CairoMakie.lines!(ax,
    df[:,5] ./ kin.T1 .- 12, df[:,7];
    linewidth   = 1.2,
    color       = 1,
    colormap    = cmap,
    colorrange  = cr,
    linestyle   = linestyles[mod1(1, length(linestyles))],
    label       = "Full WL"
)

l3 = CairoMakie.lines!(ax,
    df5[:,5] ./ kin.T1 .- 12, df5[:,7];
    linewidth   = 1.2,
    color       = 5,
    colormap    = cmap,
    colorrange  = cr,
    linestyle   = linestyles[mod1(2, length(linestyles))],
    label       = "Hoerner"
)

# l2 = CairoMakie.hlines!(ax,
#     df1[:,2][end] / ( (24/302^0.7) * 0.5 * π*(32)^2 .* 0.25^2),
#     df2[:,2][end] / ( (24/302^0.7) * 0.5 * π*(32)^2 .* 0.50^2),
#     df3[:,2][end] / ( (24/302^0.7) * 0.5 * π*(32)^2 .* 0.75^2);
#     linewidth   = 1.2,
#     color       = 5,
#     colormap    = cmap,
#     colorrange  = cr,
#     linestyle   = linestyles[mod1(2, length(linestyles))],
#     label       = L"$u_{peak}$"
# )

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

## Add a box in graph trial
savepath        = "Data/Figures/results/long_gliding_forces_plot.pdf"
figwidth_pt     = 452.97                            # Latex \textwidth in pt from the log
aspect          = 0.62                              # Ratio of height over width
figheight_pt    = figwidth_pt * aspect              # Figure height
doc_fontsize_pt = 12.5                              # Latex font size

set_theme!(Theme(font = "TeX Gyre Pagella",   fontsize = doc_fontsize_pt,))
fig             = Figure(size=(figwidth_pt, figheight_pt))
ax              = jelly_axis(fig, doc_fontsize_pt;                   # Adjust axis labels
    xlabel = L"$\phi$ [-]",
    ylabel = L"$f/(UD)^2$ [-]"
)
CairoMakie.xlims!(0,6)
CairoMakie.ylims!(-2,2)
linestyles      = [:solid, :dash, :dashdot, :dot]   # Different linestyle per line
cr              = (1,5) #extrema(Ds)                # linear range for the mapping
cmap            = :redsblues                        # pick any Makie colormap symbol

l1 = CairoMakie.lines!(ax,
    df[:,5] ./ kin.T1 .- 12, Fp ./ 64^2;
    linewidth   = 1.2,
    color       = 1,
    colormap    = cmap,
    colorrange  = cr,
    linestyle   = linestyles[mod1(1, length(linestyles))],
    label       = L"$pressure$"
)

l2 = CairoMakie.lines!(ax,
    df[:,5] ./ kin.T1 .- 12, df[:,2] ./ 64^2;
    linewidth   = 1.2,
    color       = 5,
    colormap    = cmap,
    colorrange  = cr,
    linestyle   = linestyles[mod1(1, length(linestyles))],
    label       = L"$viscous$"
)

l2 = CairoMakie.lines!(ax,
    df[:,5] ./ kin.T1 .- 12, df[:,2] ./ 64^2;
    linewidth   = 1.2,
    color       = 3,
    colormap    = cmap,
    colorrange  = cr,
    linestyle   = linestyles[mod1(1, length(linestyles))],
    label       = L"$viscous$"
)


# --- choose a zoom window (adapt to what you want to highlight) ---
xzoom = (2.0, 6.0)
yzoom = (-0.05, 0.05)

# --- inset axis (overlaid in the same grid cell as ax) ---
ax_in = CairoMakie.Axis(
    fig[1, 1],
    bbox = Makie.BBox(0.58, 0.98, 0.18, 0.55),
    xlabelvisible = false,
    ylabelvisible = false,
    xticklabelsvisible = true,
    yticklabelsvisible = true,
    backgroundcolor = :white
)

# Plot the same signals in the inset
CairoMakie.lines!(ax_in,
    df[:,5] ./ kin.T1 .- 12, Fp ./ 64^2;
    linewidth = 1.0, color = 1, colormap = cmap, colorrange = cr, linestyle = linestyles[1]
)
CairoMakie.lines!(ax_in,
    df[:,5] ./ kin.T1 .- 12, df[:,2] ./ 64^2;
    linewidth = 1.0, color = 5, colormap = cmap, colorrange = cr, linestyle = linestyles[1]
)

CairoMakie.xlims!(ax_in, xzoom...)
CairoMakie.ylims!(ax_in, yzoom...)

# --- optional: draw the zoom-rectangle on the main axis ---
zoom_rect = Rect(xzoom[1], yzoom[1], xzoom[2]-xzoom[1], yzoom[2]-yzoom[1])
CairoMakie.poly!(ax, zoom_rect; color = (:transparent), strokecolor = :black, strokewidth = 1.0)

# (optional) bring inset to front
CairoMakie.translate!(ax_in.scene, 0, 0, 10)

save(savepath, fig; pt_per_unit=1)