function gen_p_plots(sim, tᵢ, Domain)
    save_dir_p = joinpath("Figures", "Pressure_check")
    isdir(save_dir_p) || mkpath(save_dir_p)

    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))

    p = Array(sim.flow.p)        
    σ = Array(sim.flow.σ)          

    p_masked = copy(p)              
    p_masked[σ .< 0] .= NaN

    Nx, Ny = size(p_masked)
    x = range(0, Domain; length = Nx) ./ sim.L
    y = range(0, Domain; length = Ny) ./ sim.L

    pressure_plot = Plots.heatmap(x, y, p_masked', aspect_ratio=1,
    xlims=(0, Domain/sim.L), ylims=(0, Domain/sim.L), c=:balance, clims=(-2, 2),
    xlabel="x", ylabel="y", title="Pressure Field")

    Plots.contour!(x,y,sim.flow.σ',levels=[0])   
    savefig(pressure_plot, joinpath(save_dir_p, "pressure_$(tᵢ).png"))
end

function gen_u_plots(sim, tᵢ, Domain)
    save_dir_p = joinpath("Figures", "Velocity_check")
    isdir(save_dir_p) || mkpath(save_dir_p)

    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))

    u = Array(.√(sim.flow.u[:,:,1].^2 .+ sim.flow.u[:,:,2].^2))
    σ = Array(sim.flow.σ)    

    Nx, Ny = size(u)
    x = range(0, Domain; length = Nx) ./ sim.L
    y = range(0, Domain; length = Ny) ./ sim.L

    u_masked = copy(u)              
    u_masked[σ .< 0] .= NaN      

    pressure_plot = Plots.heatmap(x,y,u_masked', aspect_ratio=1,
    xlims=(0, Domain/sim.L), ylims=(0, Domain/sim.L), c=:balance, clims=(-2, 2),
    xlabel="x", ylabel="y", title="Velocity Field")

    Plots.contour!(x,y,sim.flow.σ',levels=[0])   
    savefig(pressure_plot, joinpath(save_dir_p, "velocity_$(tᵢ).png"))
end

function gen_n_plots(sim, tᵢ)
    save_dir = joinpath("Figures", "Normals_check")
    isdir(save_dir) || mkpath(save_dir)

    x = range(0, 130; length=130)
    y = range(0, 130; length=130)
    nx, ny = length(x), length(y)

    # Arrays for plotting (only nonzero vectors)
    xs, ys, nxs, nys = Float64[], Float64[], Float64[], Float64[]
    p_masked = fill(NaN, nx, ny)
    for j in 1:ny, i in 1:nx
        nvec = WaterLily.nds(sim.body, SVector(x[i], y[j]), 0.0)
        if norm(nvec) > 1e-6       # skip zero (or near-zero) vectors
            push!(xs, x[i])
            push!(ys, y[j])
            push!(nxs, nvec[1])
            push!(nys, nvec[2])
            p_masked[i,j] = sim.flow.p[i,j]
        end
    end

    fig = Figure(resolution=(700,700))
    ax = Axis(fig[1,1], title="Surface Normals (nds)", aspect=DataAspect())

    arrows!(ax, xs, ys, nxs, nys, arrowsize=10, lengthscale=3, color=:blue)
    save(joinpath(save_dir, "nds_frame_$(tᵢ).png"))
end

function gen_ω_gif(sim, t, Domain)
    save_dir_ω = joinpath("Figures", "Vorticity_check")
    isdir(save_dir_ω) || mkpath(save_dir_ω)
    @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L/sim.U
    @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I]) < 0.001, 0.0, sim.flow.σ[I])
    ω = Array(sim.flow.σ)

    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
    σ = Array(sim.flow.σ)
    ω_masked = copy(ω)
    ω_masked[σ .< 0] .= NaN

    # vorticity_plot = flood(sim.flow.σ[R] |> Array; clims=(-1, 1))
    vorticity_plot = WaterLily.flood(ω_masked,clims=(-5,5),
              cfill=:seismic,legend=false,border=:none, xlims=(0, Domain),ylims=(0, Domain),
              xlabel="x", ylabel="y", title="Vorticity at tU/D=$(round(t, digits=4))")

    vorticity_plot = Plots.contour!(sim.flow.σ',levels=[0])
    savefig(vorticity_plot, joinpath(save_dir_ω, "vorticity_$(t).png"))
end

function gen_div_gif(sim, t, R)
    @inside sim.flow.σ[I] = WaterLily.div(I,sim.flow.u)
    @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I]) < 0.001, 0.0, sim.flow.σ[I])
    WaterLily.flood(sim.flow.σ[R] |> Array; clims=(-0.5, 0.5))
    WaterLily.measure_sdf!(sim.flow.σ,sim.body,WaterLily.time(sim))
    Plots.contour!(sim.flow.σ',levels=[-sim.ϵ,0,sim.ϵ], title="$(round(t, digits=4))")
end

function gridsize_on_flap(pathing, Ncps, D, Domain)
    x = range(0,Domain;step=1); y = range(0,Domain;step=1)
    plt = Plots.plot(legend=false, title="Grid Size on flap", xlabel="x [grid cells]", ylabel="y [grid cells]")
    for xi in x
        Plots.plot!([xi, xi], [first(y), last(y)], color=:gray, lw=1)
    end
    for yi in y
        Plots.plot!([first(x), last(x)], [yi, yi], color=:gray, lw=1)
    end

    Plots.plot!(cps_at_time(pathing, 2*Ncps-1, 0)[1,:].*D,cps_at_time(pathing, 2*Ncps-1, 0)[2,:].*D, xlims=(0.85*D,1.25D), ylims=(0.15D,D/2), 
    lw=2, color=:red)

    display(plt)
end

function signed_distance_field(deg, D, Re, U, Domain, Uff, ϵ, pathing, period, period_fr; tstart=0, tend=period, step=0.1)
    save_dir = joinpath(pwd(), "Figures/", "Shape_and_SDF_Studies/")
    isdir(save_dir) || mkpath(save_dir)
    xloc = Domain / 6; yloc = Domain/4; Domain_y = Domain
    xs = range(0, Domain, step=1)
    ys = range(0, Domain_y, step=1)
    times = range(tstart, tend, step=step) 
    frames = period_fr / period .* times
    @show frames
    for fr in frames
        ν           =   U * D / Re
        cps         =   cps_at_time(pathing, 2*Ncps-1, fr) .* D .+ SA{T}[xloc, yloc]
        weights     =   ones(T, size(cps, 2)); knots       =   Float64.(knots_vector(deg, size(cps, 2))); curve       =   NurbsCurve(cps, knots, weights )
        body        =   DynamicNurbsBody(curve; thk=0, boundary=true)
        sim         =   BiotSimulation((Domain, Domain_y), (Uff,Uff), D; U, ν, body, T, mem=Array, ϵ)

        Z           =   [sdf(body, WaterLily.loc(0, CartesianIndex(x, y), eltype(sim.flow.σ)), fr) for y in ys, x in xs]
        sdf_plot    =   Plots.heatmap(xs, ys, Z; color=:viridis, xlims=(0,Domain),ylims=(0,Domain_y), aspect_ratio=1, title="Signed Distance Field $fr with deg $deg")
        Plots.contour!(xs, ys, Z, linewidth=2, color=:leonardo, levels=[-ϵ,0,ϵ])  # Contour where sdf=0
        savefig(sdf_plot, joinpath(save_dir, "sdf_t=$(fr)_deg=$(deg)_D=$(D).png"))
    end
end

function create_gif_from_folder(folder_path::String, output_path::String; delay::Float64=0.1)
    image_files = sort(filter(f -> any(ext -> endswith(lowercase(f), ext), [".png", ".jpg", ".jpeg"]), readdir(folder_path, join=true)))

    function extract_float(path)
        m = match(r"([0-9]+(?:\.[0-9]+)?) (?= \.\w+$)"x, path)
        return m === nothing ? Inf : parse(Float64, m.captures[1])
    end
    sorted_files = sort(image_files, by=extract_float)

    frames = [load(f) for f in sorted_files]

    save(output_path, cat(frames...; dims=3); fps=1/delay)
    println("GIF saved to: $output_path")
end

function generate_kin_checks(pathing, t_fr_ratio, Ncps; contr=true)
    time_range      = range(0,1.0, length=11)
    fr_range        = time_range .* t_fr_ratio 
    colors          = [:black, :red, :green, :blue, :black, :red, :green, :blue, :purple, :lightblue,:black, :red, :green, :blue, :black]

    if contr == true
        img             = load("Data/Validation_Data/Kinematic_plots.png")
        img_makie       = reverse(permutedims(img, (2, 1)), dims=2) 
        plot_steps      = 1:4
    else
        img             = load("Data/Validation_Data/Kinematic_plots2.png")
        img_makie       = reverse(permutedims(img, (2, 1)), dims=2) 
        plot_steps      = 5:10
    end


    fig             = GLMakie.Figure(resolution=(900, 600))
    ax              = GLMakie.Axis(fig[1, 1], aspect=DataAspect(), xlabel="x", ylabel="y")

    GLMakie.xlims!(ax, 0, 1.4)
    GLMakie.ylims!(ax, 0, 1.0)

    for i in plot_steps
        empty!(ax) 

        GLMakie.image!(ax, 0..1.4, 0..1, img_makie)

        fr = fr_range[i]
        cps_sm = cps_at_time(pathing, 2*Ncps-1, fr) .* 1.25

        xs_sm = Float32.(cps_sm[1, :])
        ys_sm = Float32.(cps_sm[2, :])

        GLMakie.lines!(ax, xs_sm, ys_sm, color=colors[i], linewidth=2, linestyle=:dot)

        save("Data/Validation_Data/Kinematics_check/frame_$(lpad(i,3,'0')).png", fig)
    end
end

function signal_plot(nd_time, signal, n_cycles; skip_period=true)
    if skip_period == true
        t_start = Int(round(length(signal) / n_cycles)) + 1         # Skip the first period.
    else
        t_start = 1
    end
    if signal == force
        display(Plots.plot(nd_time[t_start:end], cumsum(force[t_start:end]), xlabel="numerical time",ylabel="force",title="Forces on Jellyfish", color=:blue, legend=:topright))
    elseif signal == force_am
        display(Plots.plot(nd_time[t_start:end], cumsum(force_am[t_start:end]), xlabel="numerical time",ylabel="force", label="Added Mass Force", color=:green))
    elseif signal == force_in
        display(Plots.plot(nd_time[t_start:end], cumsum(force_in[t_start:end]), xlabel="numerical time",ylabel="force", label="Inertial Force", color=:orange))
    elseif signal == force_dr
        display(Plots.plot(nd_time[t_start:end], cumsum(force_dr[t_start:end]), xlabel="numerical time",ylabel="force", label="Drag Force", color=:red))
    elseif signal == velocity
        display(Plots.plot(nd_time[t_start:end], velocity[t_start:end], xlabel="numerical time", ylabel="velocity", title="Jellyfish Velocity", legend=:false))
    elseif signal == acceleration
        display(Plots.plot(nd_time[t_start:end], acceleration[t_start:end], xlabel="numerical time",ylabel="acceleration",title="Jellyfish Acceleration", legend=:false))
    elseif signal == displacement
        display(Plots.plot(nd_time[t_start:end], displacement[t_start:end], xlabel="numerical time",ylabel="displacement",title="Jellyfish Displacement", legend=:false))
    elseif signal == enstrophy
        display(Plots.plot(nd_time[t_start:end], enstrophy[t_start:end], xlabel="numerical time", ylabel="enstrophy",title="Flow Enstrophy"))
    end
end