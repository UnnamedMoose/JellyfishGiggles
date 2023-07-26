using Plots
using Images
using LinearAlgebra
using StaticArrays
using Interpolations
using BasicBSpline

# Load measurement data from DOI 10.1007/s10750-008-9589-4 displaying outer (exumbrellar)
# and inner (subumbrellar) surface of the bell of scyphomedusa (Aurelia sp.).
im = load("./dataset_01_medusae/Bajacaret2009_fig8_bellThickness.PNG")
plot(im)

x0 = [120, 20]
scale_x = 430
scale_y = 335

plot!([x0[1], x0[1]], [x0[2], x0[2]+scale_y], lw=2, label="")
plot!([x0[1], x0[1]+scale_x], [x0[2], x0[2]], lw=2, label="")

xu = [85, 100, 120, 200, 250, 320]
yu = [0, 105, 150, 225, 265, 290]
plot!(xu.+x0[1], yu.+x0[2], c=:blue)

xl = [85, 100, 120, 200, 250, 320]
yl = [0, 60, 100, 150, 170, 180]
plot!(xl.+x0[1], yl.+x0[2], c=:green)

savefig("outputs/plot_01_bellShapeDetailed.png")


