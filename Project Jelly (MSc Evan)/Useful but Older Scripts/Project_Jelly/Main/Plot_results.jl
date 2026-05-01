using CSV
using DataFrames
using Plots
using Images, ImageMagick, ImageIO
using Interpolations
using WaterLily
using Statistics
using StaticArrays
using Dierckx

Tp = Float64; T = Float64

include("Validation_arrays.jl")
include("Background_functions.jl")

folder = "Convergence_Study_D/D=128/"
df = CSV.read(joinpath(folder,"results.csv"), DataFrame)

cycles = 25; D = 2^7; Re = 302; U = 1
forces = df[:,1] ./ (0.5*D*U^2); force_am = df[:,2]./ (0.5*D*U^2); force_in = df[:,3]./ (0.5*D*U^2); force_dr = df[:,4]./ (0.5*D*U^2); time = df[:,5]; displacement = df[:,6]./D; velocity = df[:,7]./U; acceleration = df[:,8]./(U^2/D); enstrophy = df[:,9]
period_idx = Int(round(length(forces) / cycles))
t₀ = 9; t₁ = 12 # Plotting range

force_plt = Plots.plot(time[period_idx+1:end], forces[period_idx+1:end], label="Total Force", xlims=(t₀, t₁), xlabel="numerical time",ylabel="total force",title="Forces on Jellyfish", color=:blue, legend=:topright)
Plots.plot!(time[period_idx+1:end], force_am[period_idx+1:end], label="Added Mass Force", color=:green)
Plots.plot!(time[period_idx+1:end], force_in[period_idx+1:end], label="Inertial Force", color=:orange)
# Plots.plot!(time[period_idx+1:end], cumsum(force_dr[period_idx+1:end]), label="Drag Force", color=:red)
display(force_plt)

display(Plots.plot(time[period_idx+1:end], velocity[period_idx+1:end], xlims=(t₀, t₁), xlabel="numerical time", ylabel="Velocity", title="Jellyfish Velocity", legend=:false))
display(Plots.plot(time[period_idx+1:end], acceleration[period_idx+1:end], xlims=(t₀, t₁), xlabel="numerical time",ylabel="acceleration",title="Jellyfish Acceleration", legend=:false))
display(Plots.plot(time[period_idx+1:end], displacement[period_idx+1:end], xlims=(t₀, t₁), xlabel="numerical time",ylabel="displacement",title="Jellyfish Displacement", legend=:false))


dt_per = 0.001; dt_win = 0.01

signal, T = make_periodic_from(CFD_vel_per, mean(CFD_vel_per[2,:]))

tsp = t₀:dt_per:t₁*T; ysp = sample_signal(signal, tsp)

Plots.plot(CFD_vel_per[1,:], CFD_vel_per[2,:] ./ mean(CFD_vel_per[2,:]))
# Plots.scatter!(0:1:length=length(CFD_vel_per[2,:]), CFD_vel_per[2,:] ./ mean(CFD_vel_per[2,:]))
Plots.plot!(0:dt_per:1, sample_signal(signal,0:dt_per:1))

window, Twin = make_window_from(exp_vel, mean(exp_vel[2,:]))

ts = t₀:dt_win:t₀+Twin; ys = sample_signal(window(t₀), ts)

Plots.plot(ts, -ys, xlims=(t₀, t₁), xlabel="tU/D", ylabel="u/U", title="Jellyfish Forward Velocity", label="Experimental Velocity", legend=:topright)
Plots.plot!(tsp, -ysp, label="CFD velocity (Sahin 2009)")
Plots.plot!(time , velocity, label="WaterLily Velocity")

create_gif_from_folder(joinpath(folder,"Velocity_check/"), joinpath(folder,"velocity_output.gif"), delay=0.05)
create_gif_from_folder(joinpath(folder,"Pressure_check/"), joinpath(folder,"pressure_output.gif"), delay=0.05)
create_gif_from_folder(joinpath(folder,"Vorticity_check/"), joinpath(folder,"vorticity_output.gif"), delay=0.05)

plot_logger(joinpath(folder,"test_psolver"))
savefig(joinpath(folder,"psolver.png"))

"""
Checking convergence is tricky. First thing I need the statistical convergence from the Python script. 
Then that will tell what data is actually useful and can be considered for the actual convergence study.
"""

folder = "Convergence_Study_D/D=16/"
signal = :enstrophy
function get_mean_val(folder, signal, tconv, D)
    df = CSV.read(
        joinpath(folder, "results.csv"),
        DataFrame;
        select = [:time, signal]
    )

    # Keep only converged portion
    converged = df.time .>= tconv

    return mean(df[converged, signal]) ./ D
end

Ds = [16, 32, 64, 128, 256]
t_conv = [12.8125, 12.875, 15.75, 15.725, 15]
means = [
    get_mean_val("Convergence_Study_D/D=$D", signal, t_conv[i], D)
    for (i, D) in enumerate(Ds)
]

## Option 1
convergence_plot = Plots.plot(
    Ds, means,
    marker=:circle,
    xscale=:log10,
    legend=:false,
    xlabel="Grid resolution (D)",
    ylabel="Scaled mean $signal ",
    title="Convergence of mean $signal"
)
savefig(convergence_plot, "Convergence_Study_D_convergence_$signal.png")

## Option 2
function get_full_val(folder, signal, tconv)
    df = CSV.read(
        joinpath(folder, "results.csv"),
        DataFrame;
        select = [:time, signal]
    )

    # Keep only converged portion
    converged = df.time .>= tconv

    return df[converged,:]
end

linestyles = [:dash, :dashdot, :dot, :dash, :dashdot]
convergence_plot_2 = Plots.plot(
    get_full_val("Convergence_Study_D/D=16", signal, t_conv[1]).time, get_full_val("Convergence_Study_D/D=16", signal, t_conv[1])[!, signal] ./ 16^2,
    xlims = (maximum(t_conv), maximum(t_conv)+1), xlabel="tU/L", ylabel="Scaled $signal", title="$signal for varying grid resolution", label="D=16",
    lw=1.5, dpi=300, legend=:outertopright, linestyle = linestyles[1]
)
Plots.plot!(
    get_full_val("Convergence_Study_D/D=32", signal, t_conv[2]).time, get_full_val("Convergence_Study_D/D=32", signal, t_conv[2])[!, signal] ./ 32^2, label="D=32", lw=1.5, dpi=300, linestyle = linestyles[1]
)
Plots.plot!(
    get_full_val("Convergence_Study_D/D=64", signal, t_conv[3]).time, get_full_val("Convergence_Study_D/D=64", signal, t_conv[3])[!, signal] ./ 64^2, label="D=64", lw=1.5, dpi=300, linestyle = linestyles[1]
)
Plots.plot!(
    get_full_val("Convergence_Study_D/D=128", signal, t_conv[4]).time, get_full_val("Convergence_Study_D/D=128", signal, t_conv[4])[!, signal] ./ 128^2, label="D=128", lw=1.5, dpi=300, linestyle = linestyles[1]
)
Plots.plot!(
    get_full_val("Convergence_Study_D/D=256", signal, t_conv[5]).time, get_full_val("Convergence_Study_D/D=256", signal, t_conv[5])[!, signal] ./ 256^2, label="D=256", lw=1.5, dpi=300, linestyle = linestyles[1]
)
savefig(convergence_plot_2, "Convergence_Study_D_full_convergence_$signal.png")