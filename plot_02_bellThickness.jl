using Plots
using Images
using LinearAlgebra
using StaticArrays
using Interpolations
using BasicBSpline

xl = [85, 100, 120, 200, 250, 320]
yl = [0, 60, 100, 150, 170, 180]
xu = [85, 100, 120, 200, 250, 320]
yu = [0, 105, 150, 225, 265, 290]

# Thickness expressed as relative to half-width of the bell
dim = (xl[end] - xl[1]) * 350
thickness = (yu - yl) .* 180 ./ dim

scatter(1 .- (xl.-xl[1]) .* 350 ./ dim, thickness, label="Data", xlabel="Bell width",
    ylabel="Bell thickness relative to bell width")

xfit = 0:0.01:1
yfit = 0.24 .* (cos.(xfit*pi/2)).^.65
plot!(xfit, yfit, label="Fit")

savefig("outputs/plot_02_bellThickness.png")

