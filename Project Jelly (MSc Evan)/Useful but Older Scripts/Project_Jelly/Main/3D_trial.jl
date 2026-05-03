include("main.jl")

function make_3D_Jellyfish(cps_start, D, Re, U, deg, ϵ, T)
    rev_map(x,t)    = SA[(x[1]), hypot(x[2], x[3]) ] 
    cps_j           = cps_start  .* D .+ SA_F32[0.5D; 0]
    degree          = deg
    n_ctrl          = size(cps_j, 2)
    weights_j       = ones(T, n_ctrl)
    knots_j         = T.(clamped_uniform_knots(degree, n_ctrl))
    curve_j         = NurbsCurve(cps_j, knots_j, weights_j)
    body            = ParametricBody(curve_j; map=rev_map, ndims=3)
    ν               = U * D / Re
    return BiotSimulation((3D, D, D), (0,0,0), D; U, ν, body, T, mem=Array, ϵ)
end

sim                 = make_3D_Jellyfish(cps_start, D, Re, U, deg, ϵ, T)

Makie.inline!(false)

begin
    # Define geometry and motion on GPU
    sim             = make_3D_Jellyfish(cps_start, D, Re, U, deg, ϵ, T)#mem=CUDA.CuArray);

    cps             = cps_at_time(pathing, 2*Ncps+5, t*(period_fr/(period))) .* D .+ SA{T}[xloc, yloc]
    d = maximum(cps[2,:]) - yloc; h = maximum(cps[1,:]) - xloc; α = (2*h / d)^1.4

    sim.sim.body    = ParametricBodies.update!(sim.sim.body, cps, sim.flow.Δt[end])
    measure!(sim)
    biot_mom_step!(sim.flow,sim.pois,sim.ω,sim.x₀,sim.tar,sim.ftar;
                    fmm=sim.fmm,udf=fall!,acceleration=SA[-a0,0.f0,0.f0],U=SA[-v0,0.f0,0.f0])

    # Create CPU buffer arrays for geometry flow viz 
    a               = sim.flow.σ
    d               = similar(a,size(inside(a))) |> Array; # one quadrant
    md              = similar(d, (1,2,2).*size(d))

    # Set up geometry viz
    geom            = geom!(md,d,sim) |> Observable;
    ω               = ω!(md, d, sim) |> Observable

    fig             = GLMakie.Figure()
    ax              = GLMakie.Axis3(fig[1, 1], aspect = :data)

    GLMakie.volume!(ax, ω;algorithm=:mip,transparency=true,alpha=0.45,colormap=:algae,colorrange=(1,10))
    GLMakie.mesh!(ax, geom, alpha=0.6, color=:red)

    fig
end

nframes = 100
@info "Generating $nframes frames..."


isdir("frames") || mkpath("frames")
for frame in 51:nframes
    @show frame

    cps = cps_at_time(pathing, 105, frame) .* D .+ SA_F32[0.5D; 0]
    sim.sim.body = ParametricBodies.update!(sim.sim.body, cps, sim.flow.Δt[end])

    sim_step!(sim,sim_time(sim)+0.05; remeasure = true);    

    geom[]  = geom!(md, d, sim)
    ω[] = ω!(md, d, sim)

    # ---- Save frame ----
    fn = @sprintf("frames/frame_%04d.png", frame)
    save(fn, fig)
end

create_gif_from_folder("frames/", "frames/output.gif", delay=0.05)