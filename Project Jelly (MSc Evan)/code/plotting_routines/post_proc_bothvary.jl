## ADD VIZ AND POSTPROCESSING FUNCTIONS
include(joinpath(@__DIR__, "..", "visualisation_algorithms.jl"))

## UPLOAD THE ACCORDING FILES
files               = [
    "gam040_T1_1_T2_050_Tg_000.csv",
    "gam040_T1_1_T2_050_Tg_025.csv",
    "gam040_T1_1_T2_050_Tg_050.csv",
    "gam040_T1_1_T2_050_Tg_075.csv",
    "gam040_T1_1_T2_050_Tg_100.csv",
    "gam040_T1_1_T2_075_Tg_000.csv",
    "gam040_T1_1_T2_075_Tg_025.csv",
    "gam040_T1_1_T2_075_Tg_050.csv",
    "gam040_T1_1_T2_075_Tg_075.csv",
    "gam040_T1_1_T2_075_Tg_100.csv",
    "gam040_T1_1_T2_100_Tg_000.csv",
    "gam040_T1_1_T2_100_Tg_025.csv",
    "gam040_T1_1_T2_100_Tg_050.csv",
    "gam040_T1_1_T2_100_Tg_075.csv",
    "gam040_T1_1_T2_100_Tg_100.csv",
    "gam040_T1_1_T2_125_Tg_000.csv",
    "gam040_T1_1_T2_125_Tg_025.csv",
    "gam040_T1_1_T2_125_Tg_050.csv",
    "gam040_T1_1_T2_125_Tg_075.csv",
    "gam040_T1_1_T2_125_Tg_100.csv",
    "gam040_T1_1_T2_150_Tg_000.csv",
    "gam040_T1_1_T2_150_Tg_025.csv",
    "gam040_T1_1_T2_150_Tg_050.csv",
    "gam040_T1_1_T2_150_Tg_075.csv",
    "gam040_T1_1_T2_150_Tg_100.csv",
    "gam040_T1_1_T2_175_Tg_000.csv",
    "gam040_T1_1_T2_175_Tg_025.csv",
    "gam040_T1_1_T2_175_Tg_050.csv",
    "gam040_T1_1_T2_175_Tg_075.csv",
    "gam040_T1_1_T2_175_Tg_100.csv",
    "gam040_T1_1_T2_200_Tg_000.csv",
    "gam040_T1_1_T2_200_Tg_025.csv",
    "gam040_T1_1_T2_200_Tg_050.csv",
    "gam040_T1_1_T2_200_Tg_075.csv",
    "gam040_T1_1_T2_200_Tg_100.csv",
]
T1 = kin.T1
cases = [
    (; γ=0.4, T1=T1, T2=0.5*T1, Tg=0.00*T1),
    (; γ=0.4, T1=T1, T2=0.5*T1, Tg=0.25*T1),
    (; γ=0.4, T1=T1, T2=0.5*T1, Tg=0.50*T1),
    (; γ=0.4, T1=T1, T2=0.5*T1, Tg=0.75*T1),
    (; γ=0.4, T1=T1, T2=0.5*T1, Tg=1.00*T1),
    (; γ=0.4, T1=T1, T2=0.75*T1, Tg=0.00*T1),
    (; γ=0.4, T1=T1, T2=0.75*T1, Tg=0.25*T1),
    (; γ=0.4, T1=T1, T2=0.75*T1, Tg=0.50*T1),
    (; γ=0.4, T1=T1, T2=0.75*T1, Tg=0.75*T1),
    (; γ=0.4, T1=T1, T2=0.75*T1, Tg=1.00*T1),
    (; γ=0.4, T1=T1, T2=1*T1, Tg=0.00*T1),
    (; γ=0.4, T1=T1, T2=1*T1, Tg=0.25*T1),
    (; γ=0.4, T1=T1, T2=1*T1, Tg=0.50*T1),
    (; γ=0.4, T1=T1, T2=1*T1, Tg=0.75*T1),
    (; γ=0.4, T1=T1, T2=1*T1, Tg=1.00*T1),
    (; γ=0.4, T1=T1, T2=1.25*T1, Tg=0.00*T1),
    (; γ=0.4, T1=T1, T2=1.25*T1, Tg=0.25*T1),
    (; γ=0.4, T1=T1, T2=1.25*T1, Tg=0.50*T1),
    (; γ=0.4, T1=T1, T2=1.25*T1, Tg=0.75*T1),
    (; γ=0.4, T1=T1, T2=1.25*T1, Tg=1.00*T1),
    (; γ=0.4, T1=T1, T2=1.5*T1, Tg=0.00*T1),
    (; γ=0.4, T1=T1, T2=1.5*T1, Tg=0.25*T1),
    (; γ=0.4, T1=T1, T2=1.5*T1, Tg=0.50*T1),
    (; γ=0.4, T1=T1, T2=1.5*T1, Tg=0.75*T1),
    (; γ=0.4, T1=T1, T2=1.5*T1, Tg=1.00*T1),
    (; γ=0.4, T1=T1, T2=1.75*T1, Tg=0.00*T1),
    (; γ=0.4, T1=T1, T2=1.75*T1, Tg=0.25*T1),
    (; γ=0.4, T1=T1, T2=1.75*T1, Tg=0.50*T1),
    (; γ=0.4, T1=T1, T2=1.75*T1, Tg=0.75*T1),
    (; γ=0.4, T1=T1, T2=1.75*T1, Tg=1.00*T1),
    (; γ=0.4, T1=T1, T2=2*T1, Tg=0.00*T1),
    (; γ=0.4, T1=T1, T2=2*T1, Tg=0.25*T1),
    (; γ=0.4, T1=T1, T2=2*T1, Tg=0.50*T1),
    (; γ=0.4, T1=T1, T2=2*T1, Tg=0.75*T1),
    (; γ=0.4, T1=T1, T2=2*T1, Tg=1.00*T1),
]

"""
Add data and plotting parameters
Plotting routine to acquire a heatmap with Tg against T2 data.
"""
sims            = load_simulation.("data/varying_T1T2_and_Tg/" .* files)
results         = run_case.(sims, cases; geom=geom)

# --- Extract data (adapt these field names to your structs) ---
df      = DataFrame(
    Tg  = getproperty.(cases, :Tg) ./ getproperty.(cases, :T1),
    T2  = getproperty.(cases, :T2) ./ getproperty.(cases, :T1),
    COT = getproperty.(results, :COT),
)

xvals   = sort(unique(df.Tg))      # x-axis: Tg
yvals   = sort(unique(df.T2))      # y-axis: T2
Z       = fill(NaN, length(yvals), length(xvals))
xind    = Dict(v => i for (i, v) in pairs(xvals))
yind    = Dict(v => i for (i, v) in pairs(yvals))
for r in eachrow(df)
    Z[yind[r.T2], xind[r.Tg]] = r.COT
end

## General plotting
# Helper function for plotting --- centres -> edges ---
function centres_to_edges(c::AbstractVector{<:Real})
    c = collect(c)
    n = length(c)
    @assert n ≥ 2 "Need at least 2 values to build edges"
    mids  = (c[1:end-1] .+ c[2:end]) ./ 2
    left  = c[1]  - (mids[1] - c[1])
    right = c[end] + (c[end] - mids[end])
    return vcat(left, mids, right)   # length n+1
end

cr              = extrema(filter(isfinite, vec(Z)))  # robust colorrange

savepath        = "figures/results/varyT2Tg_COT_heatmap.pdf"
fig             = Figure(size = (figwidth_pt, figheight_pt))
ax              = CairoMakie.Axis(fig[1, 1];
xlabel          = L"$T_g$ [s]",
ylabel          = L"$T_2$ [s]",
)
xedges          = centres_to_edges(xvals)  # Tg edges (len = length(xvals)+1)
yedges          = centres_to_edges(yvals)  # T2 edges (len = length(yvals)+1)
# --- transpose for CairoMakie PDF heatmap pipeline ---
Zt = permutedims(Z)  # or transpose(Z), but permutedims is safest for non-Adjoint
@show length(xedges) length(yedges) size(Zt) # sanity check (run once)

hm = CairoMakie.heatmap!(ax, xedges, yedges, Zt;
    colormap = cmap,
    colorrange = cr,
    interpolate = false
)
# ===== Option A: Colorbar on the right =====
cb = CairoMakie.Colorbar(fig[1, 2], hm;
    label = L"$\mathrm{COT}$ [-]",
    labelsize = doc_fontsize_pt,
    ticklabelsize = doc_fontsize_pt,
)
colgap!(fig.layout, 10)

# ===== Option B: Colorbar on the bottom (comment Option A and use this) =====
# cb = Colorbar(fig[2, 1], hm;
#     vertical = false,
#     label = L"$\mathrm{COT}$ [-]",
#     labelsize = doc_fontsize_pt,
#     ticklabelsize = doc_fontsize_pt,
# )
# rowgap!(fig.layout, 2)
# rowsize!(fig.layout, 2, Relative(0.10))

save(savepath, fig; pt_per_unit = 1)