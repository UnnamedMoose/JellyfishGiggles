# include("JellyfishGeometry.jl")

# include("InterpolationFunctions.jl")

# include("Metrics.jl")

# include("SimulationSetup.jl")

using ParametricBodies, StaticArrays

using Images, ImageMagick, ImageIO

using Plots
# plotlyjs()  # for interactive 3D rotation (or use pyplot(), gr(), etc.)

# using GLMakie

function create_gif_from_folder(folder_path::String, output_path::String; delay::Float64=0.1)
    # Collect all image file paths with supported extensions
    image_files = sort(filter(f -> any(ext -> endswith(lowercase(f), ext), [".png", ".jpg", ".jpeg"]), readdir(folder_path, join=true)))

    # Load images
    frames = [load(f) for f in image_files]

    # Save as GIF
    save(output_path, cat(frames...; dims=3), fps=1/delay)
    println("GIF saved to: $output_path")
end

# Example usage:
# create_gif_from_folder("Normals_check/", "output.gif", delay=0.05)
create_gif_from_folder("Pressure_check/", "pressure_output.gif", delay=0.05)
# create_gif_from_folder("jl_mxNaEz/", "Older_Jellyfish.gif", delay=0.05)

# ints = [1, 2, 3, 4, 5, 6, 7]
# colors = [:red, :green, :blue, :orange, :purple, :brown, :pink, :cyan]
# alphas = []
# for i in ints
#     α = π/128 * i
#     push!(alphas, α)  
# end

# full_cps_set, axi_cps_set = create_cps_list(Tp)

# ThreeD_cps_list = Vector{Vector{Matrix{Float64}}}()

# curves_trial = []

# for (i, cps) in enumerate(axi_cps_set)
#     ThreeD_cps = []
#     trials = []
#     T = Float32
#     for alpha in alphas
#         cps_3D = zeros(3, size(cps, 2))
#         cps_3D[1,:] .= cps[1,:]
#         cps_3D[2,:] .= cps[2,:] .* cos(alpha)   
#         cps_3D[3,:] .= cps[2,:] .* sin(alpha)
#         cps_3D = SMatrix{3, 50}(cps_3D)
#         n_ctrl = size(cps_3D, 2)
#         weights = ones(T, n_ctrl)
#         knots = T.(clamped_uniform_knots(2, n_ctrl))
#         trial = NurbsCurve(T.(cps_3D), knots, weights)
#         # plot(trial, n=200, lw=2, color=:blue, title="trial", xlabel="x", ylabel="y", zlabel="z", legend=false, size=(800,600))

#         push!(ThreeD_cps, cps_3D)
#         push!(trials, trial)
#     end
#     push!(ThreeD_cps_list, ThreeD_cps)
#     push!(curves_trial, trials)
# end

# # Create a 3D plot
# plt_3D = plot(title="1/8 Axisymmetric Slice (Rotation about x-axis)",
#            xlabel="x", ylabel="y", zlabel="z", zlims=(0.0, 0.6), legend=false, size=(800,600))

# # Plot the original 2D profile (in x-y plane)
# # plot!(plt_3D, axi_cps_set[1][1,:], axi_cps_set[1][2,:], zeros(size(axi_cps_set[1][2,:])), lw=2, color=:green, label="2D Profile", add_cp=false)

# for i in 1:length(curves_trial[1])
#     @show i
#     plot!(plt_3D, curves_trial[1][i], lw=2, color=colors[i])
# end
# # display(plt_3D)

function compute_r(axi_cps_list)
    cyl_cps_list = []
    xyz_cps_list = []
    for cps_set in axi_cps_list
        # @show typeof(cps_set)
        # cyl_cps = zeros(3, length(axi_cps_list[1][1])) 
        # @show cps, typeof(cps)
        x = cps_set[1,:]

        r = cps_set[2,:]
        θ = zeros(size(r))  # Angle θ = 0 for the slice in the x-y plane
        xyz_cps = [x; r .* cos.(θ); r .* sin.(θ)]
        cyl_cps = hcat(x, r, θ)'
        # @show cyl_cps

        push!(cyl_cps_list, cyl_cps)
        push!(xyz_cps_list, xyz_cps)
    end
    return cyl_cps_list, xyz_cps_list
end

# cyl_cps_list, xyz_cps_list = compute_r(full_cps_set)

# cps_set = cyl_cps_list[1]

# plt = plot(cps_set[1,:], cps_set[2,:], cps_set[3,:], lw=2, color=:blue, title="Cylindrical Coordinates (x, r, θ)", xlabel="x", ylabel="y", zlabel="z",
# zlims=(-0.1, 0.6), legend=false)
# display(plt)

# Number of angular steps for revolution
# nθ = 10
# θ = range(0, π/4; length=nθ)

# # Generate meshgrid of θ and x
# X = repeat(cps_set[1,:]', nθ, 1)         # size: nθ × N
# R = repeat(cps_set[2,:]', nθ, 1)         # size: nθ × N
# Θ = repeat(θ, 1, length(cps_set[1,:]))   # size: nθ × N

# # Convert cylindrical to Cartesian
# Y = R .* cos.(Θ)
# Z = R .* sin.(Θ)

# # Plot the surface of revolution
# Plots.surface(X, Y, Z,
#     xlabel = "x",
#     ylabel = "y",
#     zlabel = "z",
#     title = "Revolved Axisymmetric Body",
#     legend = false,
#     aspect_ratio = :equal
# )

# Tp = Float32
# n_ctrl = size(axi_cps_set[1], 2)
# weights = ones(T, n_ctrl)
# knots = T.(clamped_uniform_knots(2, n_ctrl))
# curve = NurbsCurve(axi_cps_set[1], knots, weights)         
# body = DynamicNurbsBody(curve; thk=thk, boundary=true)

# function sdf_3d(body, p::SVector{3, T}) where T
#     x, y, z = p
#     r = sqrt(y^2 + z^2)
#     return sdf(body, SA[x, r])
# end

# GLMakie.activate!()



function visualize_cylinder(cps=axi_cps_set)
    # Geometry parameters
    # L = 2^p
    # center = SA[0, 0, 0]
    # r = L / 2
    # Cylinder surface mesh

    x = axi_cps_set[1][1,:]
    y = axi_cps_set[1][2,:]
    θ₁, θ₂ = 0, π/2
    θ = range(θ₁, θ₂, length=50)
    X = [xx for _ in θ, xx in x]
    Y = [yy * cos(t) for t in θ, yy in y]
    Z = [yy * sin(t) for t in θ, yy in y]
    # x = range(-4L, 4L, length=80)
    # Y = [xx for _ in θ, xx in x]
    # Z = [zz for _ in θ, zz in z]

    # Domain limits
    domain_x = (-4, 4)
    domain_y = (-3, 3)
    domain_z = (-4, 4)

    # Build figure and 3D axis
    fig = GLMakie.Figure(resolution=(800, 600))
    ax = GLMakie.Axis3(fig[1, 1]; title="3D Jellyfish Geometry",
                       xlabel="x", ylabel="y", zlabel="z")

    # Plot cylinder
    GLMakie.surface!(ax, X, Y, Z, color=:dodgerblue, transparency=true)
    GLMakie.wireframe!(ax, X, Y, Z, color=:black, linewidth=0.5)

    # Draw bottom rectangle (domain base)
    pts = [
        GLMakie.Point3f(domain_x[1], domain_y[1], domain_z[1]),
        GLMakie.Point3f(domain_x[2], domain_y[1], domain_z[1]),
        GLMakie.Point3f(domain_x[2], domain_y[2], domain_z[1]),
        GLMakie.Point3f(domain_x[1], domain_y[2], domain_z[1]),
        GLMakie.Point3f(domain_x[1], domain_y[1], domain_z[1])
    ]
    GLMakie.lines!(ax, pts; color=:gray, linewidth=1)

    ref_x, ref_y, ref_z = 0, 50, 25
    GLMakie.scatter!(ax, [ref_x], [ref_y], [ref_z], color=:red, markersize=20)
    GLMakie.text!(ax, [ref_x], [ref_y], [ref_z + 5], text=["Ref point"], color=:red)

    # Apply axis limits
    GLMakie.xlims!(ax, domain_x)
    GLMakie.ylims!(ax, domain_y)
    GLMakie.zlims!(ax, domain_z)

    GLMakie.display(fig)
    return fig
end

# Run it
# visualize_cylinder()

# function generate_sdf_plots(body)
#     xs = range(-3, 3, length=200)
#     ys = range(0, 1.6, length=200)
#     zs = range(0, 1.6, length=200)
#     Z = [sdf_3d(body, SA[x, y, z]) for z in zs, y in ys, x in xs]
#     @show typeof(Z)
#     signed_df = heatmap(xs, zs, Z; color=:viridis, aspect_ratio=1, title="Signed Distance Field")
#     display(signed_df)
# end

# generate_sdf_plots(body)


# """
# The body is basically:
# 2D control points:
# cps_0 = SA{T}[0.000  0.024  0.064  0.175  0.323  0.481  0.643  0.798  0.996  1.191  1.228  1.201  1.171  1.178  1.154  0.804  0.606  0.475  0.397  0.340  0.323  
#                 0.000  0.193  0.333  0.475  0.578  0.623  0.627  0.614  0.571  0.524  0.484  0.213  0.216  0.412  0.456  0.451  0.420  0.348  0.266  0.151 0.000  ] #*L .+ SA{T}[2L,3L]

# It can be exported to 3D so that:

# In cylindrical coordinates [x = constant, r = constant, θ = variable]

# Or:

# In xyz coordinates [x = constant, y = ycos(θ), z = ysin(θ)]

# It must be possible to make a parametric body and sdf for this geometry, but how?:

# I have this for my 2D SDF

#     function generate_sdf_plots(new_cps_list, thk=2.0, D=2^7, Tp=Float32, degree=3)
#     save_dir = joinpath(pwd(), "SDF_plots")
#     isdir(save_dir) || mkpath(save_dir)
#     for (i, cps) in enumerate(new_cps_list)
#         Tp = Float32
#         n_ctrl = size(cps, 2)
#         weights = ones(T, n_ctrl)
#         knots = T.(clamped_uniform_knots(degree, n_ctrl))
#         curve = NurbsCurve(cps .* 2D .+ SA{Tp}[D,3*D], knots, weights)         
#         body = DynamicNurbsBody(curve; thk=thk, boundary=true)

#         xs = range(0, 6.25 * D, length=200)
#         ys = range(0, 6.25 * D, length=200)
#         Z = [sdf(body, SA[x, y]) for y in ys, x in xs]

#         signed_df = heatmap(xs, ys, Z; color=:viridis, aspect_ratio=1, title="Signed Distance Field $i")
#         contour!(xs, ys, Z, levels=[0.0], linewidth=2, color=:red)  # Contour where sdf=0
#         # plot!(body.curve, shift=(0.5, 0.5), alpha=0.8, add_cp=true)
#         display(signed_df)
#         savefig(signed_df, joinpath(save_dir, "sdf_nurbs_$(i).png"))
#     end
# end


# """