using Printf
using Plots
using Images
using LinearAlgebra
using StaticArrays
using Interpolations
using ColorSchemes
using Optim
using DelimitedFiles

using ParametricBodies
# include("D:/git/ParametricBodies.jl/src/NurbsCurves.jl")

#include("./src/JellyfishPhysics.jl")
#using .JellyfishPhysics

include("./src/splines.jl")
include("./src/kinematics.jl")

# ===
# Test splines
s = 0:0.01:1

# ===
# Bell shape parameters - each row is for a separate segment, three snapshots measured in total.
# Digitised from Costello plot.

thetas = hcat(
    [0., 0., 0.],
    [0., 0., 0.],
    [0.390, 0.11, 0.37],
    [0.400, 0.731, 0.920],
    [0.950, 1.75, 1.44],
    [2.200, 0.125, 2.100],
)'

lengths = hcat(
    [0.020, 0.013, 0.025],
    [0.040, 0.026, 0.0515],
    [0.12, 0.11, 0.14],
    [0.194, 0.250, 0.194],
    [0.228, 0.179, 0.228],
    [0.250, 0.165, 0.250],
)'

halfThicknesses = hcat(
    [0.097, 0.099, 0.090],
    [0.097, 0.099, 0.090],
    [0.083, 0.102, 0.080],
    [0.063, 0.071, 0.049],
    [0.0335, 0.042, 0.0345],
    [0.011, 0.011, 0.011],
)'

#timeVals = [0, 0.01, 0.2, 0.4, 0.95, 1.0]
# timeVals = [0.04836558, 0.22848566, 0.41027352, 1.39926618, 1.57771848, 1.76617745];
timeVals = [0, 0.13, 0.27]

# Function for smoothing the shape parameters. Used for plotting only.
function smoothShapeParams(iSeg, cps_arr, s)
    # Pick control points from the array and make a spline
    cps_y = cps_arr[[1, iSeg+1], :]
    # pu = old_evaluate_spline(cps_y, s)
    pu = pbSpline(cps_y, s)
    return cps_y, pu
end

color_range = [RGB(get(ColorSchemes.algae, k)) for k in range(0, 1, length=kinematics_0_baseline.Nsegments)];

# ===
p = plot(dpi=200, xlabel="t/T", ylabel="Segment thickness")
# Ignore the time values in the loop.
for i in 1:kinematics_0_baseline.Nsegments
    cps, pu = smoothShapeParams(i, kinematics_0_baseline.cps_halfThicknesses, 0:0.01:1.0)
    plot!(pu[1, :], pu[2, :], label="", linewidth=3, color=color_range[i])
    plot!(cps[1, :], cps[2, :], linestyle=:dashdot, marker=:square, label="", color=color_range[i])
    plot!(vcat(timeVals, 1.0), vcat(halfThicknesses[i, :], halfThicknesses[i, 1]),
        linestyle=:dash, marker=:circle, label="", color=:red, markersize=5)
end

for i in 1:6
    filename = @sprintf("./dataset_01_medusae/smoothedShapeParams_segment_%d_Costello2020.txt", i-1)
    refData = readdlm(filename)
    plot!(refData[:, 1], refData[:, 3], linewidth=2, color=:black, linestyle=:dash, label="")
end

savefig("outputs/plot_01_bellShape_segmentParams_thickness.png")
plot!(show=true)

# ===
p = plot(dpi=200, xlabel="t/T", ylabel="Segment length")

for i in 1:kinematics_0_baseline.Nsegments
    cps, pu = smoothShapeParams(i, kinematics_0_baseline.cps_lengths, 0:0.01:1.0)
    plot!(pu[1, :], pu[2, :], label="", linewidth=3, color=color_range[i])
    plot!(cps[1, :], cps[2, :], linestyle=:dashdot, marker=:square, label="", color=color_range[i])
    plot!(vcat(timeVals, 1.0), vcat(lengths[i, :], lengths[i, 1]),
        linestyle=:dash, marker=:circle, label="", color=:red, markersize=5)
end

for i in 1:6
    filename = @sprintf("./dataset_01_medusae/smoothedShapeParams_segment_%d_Costello2020.txt", i-1)
    refData = readdlm(filename)
    plot!(refData[:, 1], refData[:, 2], linewidth=2, color=:black, linestyle=:dash, label="")
end

savefig("outputs/plot_01_bellShape_segmentParams_length.png")
plot!(show=true)

# ===
p = plot(dpi=200, xlabel="t/T", ylabel="Segment angle")

for i in 1:kinematics_0_baseline.Nsegments
    cps, pu = smoothShapeParams(i, kinematics_0_baseline.cps_thetas, 0:0.01:1.0)
    plot!(pu[1, :], pu[2, :], label="", linewidth=3, color=color_range[i])
    plot!(cps[1, :], cps[2, :], linestyle=:dashdot, marker=:square, label="", color=color_range[i])
    plot!(vcat(timeVals, 1.0), vcat(thetas[i, :], thetas[i, 1]),
        linestyle=:dash, marker=:circle, label="", color=:red, markersize=5)
end

for i in 1:6
    filename = @sprintf("./dataset_01_medusae/smoothedShapeParams_segment_%d_Costello2020.txt", i-1)
    refData = readdlm(filename)
    plot!(refData[:, 1], refData[:, 4], linewidth=2, color=:black, linestyle=:dash, label="")
end

savefig("outputs/plot_01_bellShape_segmentParams_angle.png")
plot!(show=true)

# Quick functionality test for the iterative t/T finding algorithm.
iSeg = 2
getSegmentPosition(iSeg, 0.21, kinematics_0_baseline.cps_halfThicknesses, printout=true)

# ===
# These are the shapes measured by Costello.
shapeData_Costello = [
[
0.090647482	0.221561043	0.090647482	0.221561043
0.096402878	0.3070171	0.069064748	0.314417411
0.105035971	0.372387115	0.076258993	0.39234301
0.148201439	0.422855284	0.110791367	0.462868298
0.217266187	0.468428473	0.139568345	0.508239037
0.289208633	0.508990998	0.182733813	0.553682079
0.364028777	0.51941723	0.246043165	0.599226348
0.424460432	0.524746032	0.320863309	0.639803333
0.49352518	0.540168468	0.381294964	0.675282889
0.574100719	0.545598496	0.430215827	0.695629225
0.677697842	0.526018582	0.490647482	0.716033404
0.752517986	0.503781497	0.553956835	0.716351542
0.814388489	0.453841148	0.608633094	0.696525794
0.861870504	0.388753118	0.663309353	0.66162467
0.89352518	0.316047865	0.723741007	0.631777593
0.890647482	0.215530892	0.784172662	0.59690539
],
[
1.092086331	0.322070786	1.092086331	0.322070786
1.126618705	0.342344818	1.107913669	0.382451827
1.169784173	0.397838111	1.14676259	0.463049058
1.21294964	0.473431908	1.195683453	0.558772279
1.263309353	0.523936228	1.264748201	0.637008785
1.310791367	0.549300459	1.353956835	0.705296266
1.379856115	0.564722895	1.44028777	0.755981346
1.454676259	0.585199378	1.500719424	0.776385525
1.543884892	0.610773291	1.552517986	0.781670945
1.610071942	0.611105889	1.620143885	0.769447959
1.722302158	0.581519106	1.69352518	0.732128267
1.797122302	0.501493077	1.75971223	0.679697046
1.837410072	0.44139402	1.817266187	0.607121941
1.860431655	0.386233325	1.85323741	0.526900691
1.883453237	0.321022378	1.871942446	0.446592676
1.915107914	0.250829688	1.883453237	0.37629876
1.915107914	0.250829688		1.902158273	0.326141499
1.915107914	0.250829688		1.915107914	0.250829688
],
[
2.237410072	0.232348794	2.237410072	0.232348794
2.231654676	0.347897762	2.205755396	0.347767615
2.223021583	0.448356892	2.197122302	0.468327248
2.243165468	0.54896063	2.202877698	0.573883808
2.274820144	0.604396081	2.231654676	0.674530928
2.335251799	0.670026391	2.271942446	0.745085138
2.381294964	0.690358266	2.322302158	0.798102021
2.438848921	0.720798236	2.385611511	0.856209103
2.530935252	0.731311232	2.454676259	0.894244604
2.6	0.706532663	2.530935252	0.907190629
2.686330935	0.681840859	2.598561151	0.89999277
2.755395683	0.606811034	2.646043165	0.875105745
2.798561151	0.526625935	2.697841727	0.832652471
2.824460432	0.441328947	2.738129496	0.782603666
2.818705036	0.35587289	2.772661871	0.737551065
2.795683453	0.250229565	2.810071942	0.672412422
2.795683453	0.250229565		2.835971223	0.592140559
2.795683453	0.250229565		2.85323741	0.511825314
2.795683453	0.250229565		2.850359712	0.436433968
2.795683453	0.250229565		2.830215827	0.340855356
2.795683453	0.250229565		2.795683453	0.250229565
],
[
0.096402878	-0.788460287	0.096402878	-0.788460287
0.079136691	-0.713170167	0.058992806	-0.718296519
0.096402878	-0.637706518	0.073381295	-0.605158888
0.139568345	-0.567137848	0.125179856	-0.486808141
0.194244604	-0.531687213	0.189928058	-0.416131015
0.269064748	-0.506185604	0.273381295	-0.36043527
0.352517986	-0.485665739	0.348201439	-0.309808033
0.433093525	-0.485260837	0.435971223	-0.269165974
0.539568345	-0.509851415	0.505035971	-0.266306352
0.614388489	-0.534601063	0.571223022	-0.281049131
0.703597122	-0.564303532	0.634532374	-0.323444561
0.795683453	-0.624142294	0.689208633	-0.380958751
0.850359712	-0.684169047	0.741007194	-0.430949713
0.861870504	-0.749437837	0.801438849	-0.488434981
0.867625899	-0.797147609	0.844604317	-0.548519576
0.867625899	-0.797147609		0.870503597	-0.631304002
0.867625899	-0.797147609		0.884892086	-0.716658834
0.867625899	-0.797147609		0.867625899	-0.797147609
],
[
1.060431655	-0.743414916	1.060431655	-0.743414916
1.10647482	-0.718057915	1.092086331	-0.68295434
1.141007194	-0.672658255	1.117985612	-0.622522685
1.178417266	-0.582018004	1.138129496	-0.54201945
1.225899281	-0.508915079	1.175539568	-0.446354073
1.276258993	-0.460923322	1.227338129	-0.365691768
1.330935252	-0.435522938	1.296402878	-0.300018076
1.4	-0.410050251	1.368345324	-0.239355049
1.469064748	-0.404678067	1.460431655	-0.183615921
1.555395683	-0.40173168	1.538129496	-0.173175229
1.644604317	-0.398770833	1.607194245	-0.182878421
1.705035971	-0.408517407	1.670503597	-0.217736163
1.75971223	-0.463519034	1.73381295	-0.277719533
1.797122302	-0.523632551	1.789928058	-0.342764181
1.817266187	-0.588857959	1.824460432	-0.420480098
1.825899281	-0.659166335	1.846043165	-0.503286215
1.84028777	-0.724420664	1.85323741	-0.59872745
1.877697842	-0.769458805	1.857553957	-0.704233397
1.877697842	-0.769458805		1.877697842	-0.769458805
],
[
2.243165468	-0.717371028	2.243165468	-0.717371028
2.246043165	-0.641979683	2.214388489	-0.622038249
2.234532374	-0.55661039	2.2	-0.531658291
2.228776978	-0.461161925	2.194244604	-0.436209826
2.254676259	-0.395705144	2.202877698	-0.350739308
2.297841727	-0.325136474	2.235971223	-0.260120748
2.349640288	-0.279650049	2.289208633	-0.184476339
2.404316547	-0.254249666	2.351079137	-0.12888905
2.473381295	-0.23882723	2.410071942	-0.088391598
2.565467626	-0.24338961	2.482014388	-0.06290445
2.700719424	-0.30301146	2.551079137	-0.062557391
2.752517986	-0.373102925	2.611510791	-0.072303966
2.789928058	-0.453316944	2.66618705	-0.102179965
2.801438849	-0.543711363	2.723741007	-0.157167131
2.784172662	-0.639275514	2.764028777	-0.212241061
2.766906475	-0.72981454	2.807194245	-0.287401034
2.766906475	-0.72981454		2.838848921	-0.372669101
2.766906475	-0.72981454		2.850359712	-0.447988142
2.766906475	-0.72981454		2.844604317	-0.533444199
2.766906475	-0.72981454		2.824460432	-0.608922309
2.766906475	-0.72981454		2.795683453	-0.67439355
2.766906475	-0.72981454		2.766906475	-0.72981454
]
];

# Function for navigating the data structure.
function shapeCostello(iCostello_a)
    x_ref_a = vcat([shapeData_Costello[iCostello_a][:, 1], shapeData_Costello[iCostello_a][:, 3]]...)
    y_ref_a = vcat([shapeData_Costello[iCostello_a][:, 2], shapeData_Costello[iCostello_a][:, 4]]...)
    x0_a = (findmax(x_ref_a)[1] + findmin(x_ref_a)[1]) / 2.0
    y0_a = findmax(y_ref_a)[1]
    return x_ref_a .- x0_a, y_ref_a .- y0_a
end

# ===
plot(xlabel="x/L", ylabel="y/L", aspect_ratio=:equal, size=(1000, 450))

# ---
xy, cps, segParams1 = shapeForTime(0., kinematics_0_baseline)
x_ref_a, y_ref_a = shapeCostello(1)
x_ref_b, y_ref_b = shapeCostello(4)

plot!(abs.(x_ref_a), y_ref_a, marker=:dot, linewidth=0, color=:red, markersize=2,
    label="Costello et al., picture 1, t/T=0.00")
plot!(abs.(x_ref_b), y_ref_b, marker=:v, linewidth=0, color=:red, markersize=2,
    label="Costello et al., picture 2, t/T=0.00")
plot!(xy[1, :], xy[2, :], linewidth=2, color=:red, label="Current model, t/T=0.00")
plot!(cps[1, :], cps[2, :], linewidth=2, color=:red, marker=:square, linestyle=:dash, markersize=4, label="")

shape_Costello_regressed = readdlm("./dataset_01_medusae/shape_Costello2020_snapshot1.txt")
refCps = readdlm("./dataset_01_medusae/smoothShapeCps_Costello2020_snapshot0.txt")
plot!(shape_Costello_regressed[:, 1], shape_Costello_regressed[:, 2], linewidth=2, color=:black, label="")
plot!(refCps[:, 1] .+ 0.0, refCps[:, 2], linewidth=2, marker=:x, linestyle=:dashdotdot, color=:black,
    markersize=5, label="")

# ---
xy, cps, segParams2 = shapeForTime(0.13, kinematics_0_baseline)
x_ref_a, y_ref_a = shapeCostello(2)
x_ref_b, y_ref_b = shapeCostello(5)

plot!(abs.(x_ref_a) .+ 0.5, y_ref_a, marker=:dot, linewidth=0, color=:green, markersize=2,
    label="Costello et al., picture 1, t/T=0.13")
plot!(abs.(x_ref_b) .+ 0.5, y_ref_b, marker=:v, linewidth=0, color=:green, markersize=2,
    label="Costello et al., picture 2, t/T=0.13")
plot!(xy[1, :] .+ 0.5, xy[2, :], linewidth=2, color=:green, label="Current model, t/T=0.13")
plot!(cps[1, :] .+ 0.5, cps[2, :], linewidth=2, color=:green, marker=:square, linestyle=:dash, markersize=4, label="")

shape_Costello_regressed = readdlm("./dataset_01_medusae/shape_Costello2020_snapshot2.txt")
refCps = readdlm("./dataset_01_medusae/smoothShapeCps_Costello2020_snapshot1.txt")
plot!(shape_Costello_regressed[:, 1], shape_Costello_regressed[:, 2], linewidth=2, color=:black, label="")
plot!(refCps[:, 1] .+ 0.5, refCps[:, 2], linewidth=2, marker=:x, linestyle=:dashdotdot, color=:black,
    markersize=5, label="")

# ---
xy, cps, segParams3 = shapeForTime(0.27, kinematics_0_baseline)
x_ref_a, y_ref_a = shapeCostello(4)
x_ref_b, y_ref_b = shapeCostello(6)

plot!(abs.(x_ref_a) .+ 1.0, y_ref_a, marker=:dot, linewidth=0, color=:blue, markersize=2,
    label="Costello et al., picture 1, t/T=0.27")
plot!(abs.(x_ref_b) .+ 1.0, y_ref_b, marker=:v, linewidth=0, color=:blue, markersize=2,
    label="Costello et al., picture 2, t/T=0.27")
plot!(xy[1, :] .+ 1.0, xy[2, :], linewidth=2, color=:blue, label="Current model, t/T=0.27")
plot!(cps[1, :] .+ 1.0, cps[2, :], linewidth=2, color=:blue, marker=:square, linestyle=:dash, markersize=4, label="")

shape_Costello_regressed = readdlm("./dataset_01_medusae/shape_Costello2020_snapshot3.txt")
refCps = readdlm("./dataset_01_medusae/smoothShapeCps_Costello2020_snapshot2.txt")
plot!(shape_Costello_regressed[:, 1], shape_Costello_regressed[:, 2], linewidth=2, color=:black, label="")
plot!(refCps[:, 1] .+ 1.0, refCps[:, 2], linewidth=2, marker=:x, linestyle=:dashdotdot, color=:black,
    markersize=5, label="")

plot!(legend=:outertop, legend_columns=3, show=true)
savefig("outputs/plot_02_bellShape_staticCostelloComparison.png")

# ===
# Function to generate frames for the animation
function generate_frames()
    anim = Animation()
    for x in 0:0.01:1
        xy, cps, segParams = shapeForTime(x, kinematics_0_baseline)
        fs = 14
        plot(xlabel="x/L", ylabel="y/L", aspect_ratio=:equal, xlims=(0, 0.75), ylims=(-0.75, 0), size=(1000, 800),
            xtickfontsize=fs, ytickfontsize=fs, xguidefontsize=fs, yguidefontsize=fs, legendfontsize=fs)

        shape_Costello_regressed = readdlm("./dataset_01_medusae/shape_Costello2020_snapshot1.txt")
        plot!(shape_Costello_regressed[:, 1], shape_Costello_regressed[:, 2], linewidth=2, color=:red,
            linestyle=:dash, label="Costello et al., t/T=0.00")

        shape_Costello_regressed = readdlm("./dataset_01_medusae/shape_Costello2020_snapshot2.txt")
        plot!(shape_Costello_regressed[:, 1].-0.5, shape_Costello_regressed[:, 2], linewidth=2, color=:green,
            linestyle=:dash, label="Costello et al., t/T=0.13")

        shape_Costello_regressed = readdlm("./dataset_01_medusae/shape_Costello2020_snapshot3.txt")
        plot!(shape_Costello_regressed[:, 1].-1.0, shape_Costello_regressed[:, 2], linewidth=2, color=:blue,
            linestyle=:dash, label="Costello et al., t/T=0.27")

        plot!(xy[1, :], xy[2, :], color=:black, linewidth=3, label="")
        plot!(cps[1, :], cps[2, :], linewidth=2, color=:black, marker=:circle, linestyle=:dash, markersize=4, alpha=0.5, label="")

        formatted_float = @sprintf("%.2f", x)
        annotate!(0.65, -0.7, text("t/T=$formatted_float", fs, :black))

        frame(anim)
    end
    return anim
end

# Generate frames
anim = generate_frames()

# Save the animation as a GIF
gif(anim, "outputs/plot_03_animatedBellShape.gif", fps=10)

