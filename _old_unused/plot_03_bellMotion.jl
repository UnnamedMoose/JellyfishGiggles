using Plots
using Images
using LinearAlgebra
using StaticArrays
using Interpolations
using BasicBSpline

bellShapeX = s -> sign(s)*(1.0f0-cos(clamp(abs(s), 0, 1)*pi/2.0f0))^0.75f0
bellShapeY = s -> (sin(pi/2.0f0 - clamp(abs(s), 0, 1)*(pi/2.0f0)))^0.5f0 - 1.0f0

s_motion = [0.42, 0.65, 0.82, 0.95]
x_motion = bellShapeX.(s_motion)
y_motion = bellShapeY.(s_motion)

# Apply motion.
# Assumption: bell is relaxad for part of the duration of the complete cycle
# (https://doi.org/10.1146/annurev-marine-031120-091442) and half a sine wave otherwise.
amplitudes = [0.02, 0.075, 0.15, 0.35]
t = 0:0.01:1
f_relax = 0.2
deformed = []
for i in range(1, length(amplitudes))
    x = []
    y = []
    for tt in t
        if tt > f_relax
            push!(x, x_motion[i] + amplitudes[i]*sin((tt-f_relax)/(1-f_relax)*pi))
            push!(y, y_motion[i] + amplitudes[i]*sin((tt-f_relax)/(1-f_relax)*pi))
        else
            push!(x, x_motion[i])
            push!(y, y_motion[i])
        end
    end
    push!(deformed, [x, y])
end

# Load measurement data from DOI 10.1007/s10750-008-9589-4.
im = load("./dataset_01_medusae/Bajacaret2009_fig6_bellMotion_cropped.PNG")
plot(im)

# Figure out plot coordinates.
x0_x = [70, 310]
scale_x = -180#-290

x0_y = [70, 660]
scale_y = -320

scale_t = 410

plot!([x0_x[1], x0_x[1]], [x0_x[2], x0_x[2]+scale_x], lw=2, label="")
plot!([x0_x[1], x0_x[1]+scale_t], [x0_x[2], x0_x[2]], lw=2, label="")
plot!([x0_y[1], x0_y[1]], [x0_y[2], x0_y[2]+scale_y], lw=2, label="")

# Plot the position of the points in the baseline underformed configuration
scatter!(zeros(length(x_motion)).+x0_x[1], x_motion.*scale_x .+ x0_x[2], label="x0")
scatter!(zeros(length(y_motion)).+x0_y[1], -y_motion.*scale_y .+ x0_y[2], label="y0")

# TODO how to plot inside a bloody loop?
plot!(t.*scale_t.+x0_x[1], deformed[1][1].*scale_x .+ x0_x[2])
plot!(t.*scale_t.+x0_y[1], -deformed[1][2].*scale_y .+ x0_y[2])

plot!(t.*scale_t.+x0_x[1], deformed[2][1].*scale_x .+ x0_x[2])
plot!(t.*scale_t.+x0_y[1], -deformed[2][2].*scale_y .+ x0_y[2])

plot!(t.*scale_t.+x0_x[1], deformed[3][1].*scale_x .+ x0_x[2])
plot!(t.*scale_t.+x0_y[1], -deformed[3][2].*scale_y .+ x0_y[2])

plot!(t.*scale_t.+x0_x[1], deformed[4][1].*scale_x .+ x0_x[2])
plot!(t.*scale_t.+x0_y[1], -deformed[4][2].*scale_y .+ x0_y[2])

savefig("outputs/plot_03_bellMotion.png")

# ===
scatter(s_motion, amplitudes, label="Fitting points",
    xlabel="s-param of bell shape", ylabel="motion amplitude")

s = 0:0.01:1
amp_fit = 0.45.*s.^4
plot!(s, amp_fit, label="Analytical amplitude profile along half of the bell")

savefig("outputs/plot_03_bellMotionAmplitudeFit.png")

