include(joinpath(@__DIR__, "..", "visualisation_algorithms.jl"))
include(joinpath(@__DIR__, "..", "..",  "data", "validation_arrays", "Sarsia_validation_arrays.jl"))

"""
Plotting Routines for:
    Kinematic algorithm check (oral cavity volume over time)
    Oral cavity volume and velar opening diameter check over time
"""

## Load the right files, either a validation against (Sahin 2009) or (Colin 2002), CFD and experimental data, respectively.
# Get Validation Data, (ts, ys) = Sahin 2009, (tsp, ysp) = Colin 2002
comp           = CSV.read("data/duty_cycle_variation_data/3D_D64_eps2_dt005_basecase_gam035.csv", DataFrame)   # Sahin 2009
t₀ = 15; t_end = 20; dt_per = 0.001; dt_win = 0.01;
# signal, T      = make_periodic_from(Sahin_vel_per);      tsp = t₀:dt_per:t_end*T;    ysp = sample_signal(signal, tsp);
signal, T      = make_periodic_from(Sahin_vel_per);  tsp = t₀:dt_per:t_end*T;    ysp = sample_signal(signal, tsp);

comp           = CSV.read("data/validation_arrays/3D_D64_Colin_rebuild.csv", DataFrame) # Colin 2002
window, Twin   = make_window_from(colin_vel);            ts  = t₀:dt_win:t₀+Twin;    ys  = sample_signal(window(t₀), ts);

## Compute additional data from the results
cps         = cps_at_time(pathing, geom.Ncps, 0) .* 64
WL_acc      = comp[:,1]  ./ shape_volume(SMatrix{2,20,Float64}(cps[:,16:end]) .* 0.85)
vel_WL      = cumsum(WL_acc .* 0.05)
Fcoef       = 2 .* comp[:,1] ./ (64^2)
a           = findfirst(>(10 .* kin.T1), comp[:,2]) -1
b           = findfirst(>(15 .* kin.T1), comp[:,2])
WL_FR       = valdat.FR .* 1.0706 / 0.96

time_ar = comp[:,2] .- 0.045
a = findfirst(>(tsp[1] .* kin.T1), time_ar) - 1
b = findfirst(>(tsp[end] .* kin.T1), time_ar) 

s1 = ysp 
itp = LinearInterpolation(time_ar[a:b], Fcoef[a:b])
s = itp.(tsp .* kin.T1)
rmse = sqrt(mean((s .- s1).^2))
ampl = maximum(s) - minimum(s)
nrmse = rmse / ampl


s1 = -ys
itp = LinearInterpolation(WL_force[:,2][a:b], WL_force[:,4][a:b])
s = itp.(ts .* kin.T1)
rmse = sqrt(mean((s .- s1).^2))
ampl = maximum(s) - minimum(s)
nrmse = rmse / ampl



## Plotting Routines
savepath        = "figures/validation/Velocity_Validation_Colin.pdf"
fig             = Figure(size=(figwidth_pt, figheight_pt))
ax              = jelly_axis(fig, doc_fontsize_pt;                   # Adjust axis labels
xlabel          = L"$\phi$ [-]",
ylabel          = L"$u_x/U$ [-]"
)

CairoMakie.xlims!(ax, 15, 20)                         # Axis limits
# CairoMakie.ylims!(ax, -1.5, -0.5)

l1 = CairoMakie.lines!(ax,
    # valdat.t, valdat.FR .* 1.0706 / 0.96 ;
    comp[:,2] ./ kin.T1, vel_WL ; 
    # comp[:,2], comp[:,1] ./ 64^2;
    linewidth   = 1.2,
    color       = 1, colormap = cmap, colorrange = cr,
    linestyle   = linestyles[mod1(1, length(linestyles))],
    label       = "WaterLily"
)

l2 = CairoMakie.lines!(ax,
    # colin_FR[1,:], colin_FR[2,:] ;
    ts, ys ./ -mean(ys);
    # tsp, ysp ./ -mean(ysp);
    linewidth   = 1.2,
    color       = 5, colormap = cmap, colorrange = cr,
    linestyle   = linestyles[mod1(2, length(linestyles))],
    label       = "Colin et al. (2002)"
    # label       = "Sahin et al. (2009)"
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