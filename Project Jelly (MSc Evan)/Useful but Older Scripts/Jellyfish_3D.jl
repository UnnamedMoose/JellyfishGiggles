@inline function ThreeDimJellyfish(::Type{T}=Float32; 
    new_cps_list, D=2^7, Re=302, U=1, ϵ=0.5, thk=2ϵ+√3, deg, 
    mem=Array, use_biotsavart=false) where {T<:AbstractFloat}

    revolve_map(x,t) = SA[x[1], hypot(x[2], x[3])]

    cps = new_cps_list[1] .* D/2 
    degree = deg
    n_ctrl = size(cps, 2)
    weights = ones(T, n_ctrl)
    knots = T.(clamped_uniform_knots(degree, n_ctrl))
    curve = NurbsCurve(cps, knots, weights)

    body = ParametricBody(curve;map=revolve_map,ndims=3)

    ν = U * D / Re

    return use_biotsavart ?
        BiotSimulation((6D, 6D, 6D), (0,0,0), D; U, ν, body, T, mem, ϵ) :
        Simulation((6D, 6D, 6D), (0,0,0), D; U, ν, body, T, mem, ϵ)
end

function visualise_3D_Jelly(sim, θ₂)
    X = [sim.body.map(sim.body.curve(u), θ)[1] for u in LinRange(0, 1, 80), θ in LinRange(0, θ₂, 80)]
    Y = [sim.body.map(sim.body.curve(u), θ)[2] for u in LinRange(0, 1, 80), θ in LinRange(0, θ₂, 80)]
    Z = [sim.body.map(sim.body.curve(u), θ)[3] for u in LinRange(0, 1, 80), θ in LinRange(0, θ₂, 80)]

    fig = Figure(resolution = (900, 700))
    ax = Axis3(fig[1, 1], title = "Revolved NURBS Jellyfish")
    GLMakie.surface!(ax, X, Y, Z, colormap = :Blues, shading = true)
    ax.xlabel = "x"
    ax.ylabel = "y"
    ax.zlabel = "z"
    fig
end

