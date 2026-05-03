include(joinpath(@__DIR__, "..", "visualisation_algorithms.jl"))
include(joinpath(@__DIR__, "..", "..",  "data", "validation_arrays", "Sarsia_validation_arrays.jl"))

"""
Plotting Routines for:
    Kinematic algorithm check (oral cavity volume over time)
    Oral cavity volume and velar opening diameter check over time
"""

## Prescribed kinematics check
savepath        = "figures/validation/prescribed_kins_check.pdf"
fig             = Figure(size=(figwidth_pt, figheight_pt))
ax              = jelly_axis(fig, doc_fontsize_pt;                   # Adjust axis labels
xlabel          = L"$\phi(T_1)$ [-]",
ylabel          = L"$V/D^3$ [-]"
)
CairoMakie.xlims!(0,2.5)

cr              = (1,5)               

cases = [
    (γ=0.4, varyingT=false, gliding=false, label=L"$base$"),
    (γ=0.2, varyingT=false, gliding=false, label=L"$\gamma$"),
    (γ=0.4, varyingT=true,  gliding=false, label=L"$T_2$"),
    (γ=0.4, varyingT=false, gliding=true,  label=L"$T_g$"),
    (γ=0.4, varyingT=true,  gliding=true,  label=L"$T_2$, $T_g$")
]

handles = Makie.AbstractPlot[]
ts = collect(0:0.01:2.5 * kin.T1)

for (i, case) in enumerate(cases)
    motion = generate_jelly_motion(
        contr_frames, exp_frames, geom.Ncps, kin.T1, 0.5*scaling.tscale, 0.5*scaling.tscale,
        geom.n_cycles, geom.n_up, case.γ;
        ThreeD=true,
        varyingT=case.varyingT,
        gliding=case.gliding
    )
    y = Vector{Float64}(undef, length(ts))

    for (j, t) in enumerate(ts)
        cps = SMatrix{2,15,Float64}(cps_at_time(motion, geom.Ncps, t)[:, 1:15])
        y[j] = shape_volume(cps)
    end
    ln = CairoMakie.lines!(
        ax,
        collect(0:0.01:2.5*kin.T1) ./ kin.T1,
        y;
        linewidth  = 1.2,
        color      = i,
        colormap   = cmap,
        colorrange = cr,
        linestyle  = linestyles[mod1(i, length(linestyles))],
        label      = case.label
    )

    push!(handles, ln)
end

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

"""
--- KINEMATIC VALIDATION OF ORAL CAVITY VOLUME AND VELAR OPENING DIAMETER ---
Can also do Mass Conservation. The valdat structure contains the data, this should be ran first. The rmse-values can also be computed here.
"""
valdat          = compute_validation_data(pathing, geom, kin; dt=0.01)
s1              = lip_vel_diam[2,:] ./ scaling.Dmax
itp             = LinearInterpolation(valdat.t, valdat.velar_diam)
s               = itp.(lip_vel_diam[1,:])
rmse            = sqrt(mean((s .- s1).^2))
ampl            = maximum(s) - minimum(s)
nrmse           = rmse / ampl

savepath        = "figures/validation/Oral_Cavity_Volume_2D_check.pdf"
fig             = Figure(size=(figwidth_pt, figheight_pt))
ax              = jelly_axis(fig, doc_fontsize_pt;                   # Adjust axis labels
xlabel          = L"$\phi$ [-]",
ylabel          = L"$V/D^3$ [-]"
)
cr              = (1,5) 

CairoMakie.xlims!(ax, 0, 2)                         # Axis limits
# CairoMakie.ylims!(ax, -2, 2)

l1 = CairoMakie.lines!(ax,
    valdat1.t ./ kin.T1, valdat1.vol_cav;
    # valdat.t, valdat.velar_diam;
    linewidth   = 1.2,
    color       = 1, colormap = cmap, colorrange = cr,
    linestyle   = linestyles[mod1(1, length(linestyles))],
    label       = "WaterLily 2D"
)

l1 = CairoMakie.lines!(ax,
    valdat2.t ./ kin.T1, valdat2.vol_cav;
    # valdat.t, valdat.velar_diam;
    linewidth   = 1.2,
    color       = 5, colormap = cmap, colorrange = cr,
    linestyle   = linestyles[mod1(2, length(linestyles))],
    label       = "WaterLily 3D"
)

l2 = CairoMakie.lines!(ax,
    lip_cav_vol[1,:], lip_cav_vol[2,:] ./ scaling.Dmax^3;
    # lip_vel_diam[1,:], lip_vel_diam[2,:] ./ scaling.Dmax;
    linewidth   = 1.2,
    color       = 2, colormap = cmap, colorrange = cr,
    linestyle   = linestyles[mod1(3, length(linestyles))],
    label       = "Lipinski et al. (2009)"
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