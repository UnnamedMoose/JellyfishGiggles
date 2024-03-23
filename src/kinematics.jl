using Plots
using Images
using LinearAlgebra
using StaticArrays
using Interpolations
using ColorSchemes
using Optim
using StaticArrays
using CUDA

struct KinematicsArray{T, N, M}
    Nsegments::Int
    cps_thetas::StaticArrays.SArray{Tuple{N, M}, T}
    cps_lengths::StaticArrays.SArray{Tuple{N, M}, T}
    cps_halfThicknesses::StaticArrays.SArray{Tuple{N, M}, T}
end


kinematics_0_baseline = KinematicsArray{Float64, 6+1, 7}(
    # No. segments
    6,
    # Segment angles.
    SMatrix{6+1, 7}(vcat([
    # t/T values
    [0, 0.025, 0.15, 0.25, 0.35, 0.45, 1.0]',
    # CP locations
    hcat([
        [0., 0.,        0., 0., 0.,       0., 0.],
        [0., 0.,        0., 0., 0.,       0., 0.],
        [0.39, 0.39,        0., 0.38, 0.39,       0.39, 0.39],
        [0.4, 0.4,        0.85, 1.2, 0.41,       0.4, 0.4],
        [0.95, 0.95,        1.95, 1.65, 0.95,       0.95, 0.95],
        [2.2, 2.2,        -1., 2.19, 2.2,       2.2, 2.2],
    ]...)'
    ]...)),
    # Segment lengths
    SMatrix{6+1, 7}(vcat([
    [0, 0.025, 0.15, 0.25, 0.35, 0.45, 1.0]',
    hcat([
        [0.02, 0.02,        0.01, 0.03, 0.022,       0.02, 0.02],
        [0.04, 0.04,        0.02, 0.06, 0.04,       0.04, 0.04],
        [0.12, 0.12,        0.1, 0.155, 0.11,       0.12, 0.12],
        [0.194, 0.194,        0.275, 0.194, 0.194,       0.194, 0.194],
        [0.228, 0.228,        0.16, 0.225, 0.228,       0.228, 0.228],
        [0.250, 0.250,        0.135, 0.245, 0.250,       0.250, 0.250],
    ]...)'
    ]...)),
    # Segment half-thicknesses
    SMatrix{6+1, 7}(vcat([
    [0, 0.025, 0.15, 0.25, 0.35, 0.45, 1.0]',
    hcat([
        [0.097, 0.097,        0.1005, 0.0875, 0.097,       0.097, 0.097],
        [0.097, 0.097,        0.1005, 0.0875, 0.097,       0.097, 0.097],
        [0.083, 0.083,        0.11, 0.075, 0.083,       0.083, 0.083],
        [0.063, 0.063,        0.075, 0.04, 0.063,       0.063, 0.063],
        [0.0335, 0.0335,        0.045, 0.0345, 0.034,       0.0335, 0.0335],
        [0.011, 0.011,        0.011, 0.011, 0.011,       0.011, 0.011],
    ]...)'
    ]...))
)

