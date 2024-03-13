# JellyfishGiggles

Jellyfish and Julia

Contents:
    - script_00_bellShapeSpline.jl test the kinematics obtained from source papers.
        They should be doing roughly the same thing, but one in Python and one in Julia.

Notes to self:
- To monitor GPU usage, use `nvidia-smi -l`

Dependencies:
    import Pkg; Pkg.add("Images"); Pkg.add("Interpolations"); Pkg.add("ColorSchemes"); Pkg.add("Optim")

TODO
x get the ParametricBodies.jl spline running and compare to my old implementation
    to make sure that for the same order and CPs we get the same spline

| update CPs and kinematics to match Python, they have been improved
    x try using NURBS with repeated knots to directly interpolate the kinematics instead of
        using the interpolation package
    - rewrite the old profileFromParams routine to match what PB uses to generate
        moving body shapes
    - compare directly to Costello data for the three available cycles to make sure
        everything's fine
    - save an animation out of julia to make sure we can generate unsteady profiles
        
| rewrite the src/JellyfishPhysics.jl to use PB.jl splines. No need to duplicate code.

- run a simple simulation with no flow to check the kinematics and SDF; start with using
    just the centreline and fixed thickness
- add the flow and run that
- make sure the code can run on the GPU node
- add proper filled 2D profile
- add B-S BC (ask Marin for help)
- add 1D motion with MRF (ask Marin for help)
- make the SDF axisymmetric in 3D for validation
- compare to validation data for real jellyfish. If it's good, write a paper!

- change motion to 2D keeping everything else the same.
- see if can exchange messages between Python and Julia - best to use stable baselines
    since I already know how to set it up and get good results with it
- Run some RL using 2D flow (3D will for sure be too expensive and the SDF won't work).
    + Start at a random position and orientation and make them swim towards a certain location.
    + Can add unknown current of variable magnitudes as well to make it harder.
    + Use incremental reward as they get closer to the target and a nice fat bonus
        at the end. Penalty for leaving a pre-defined domain.
    + Later on can think/read about energy expenditure for a particular motion
        and see if we can add that as a rolling penalty.
    + Compare resultant kinematics to what the real animal does.
    + Write a paper!
- if it all works fine, ask Gabe and Marin how we could make the SDF 3D properly
    so that we could at least test the 2D kinematics in a much more realistic
    setting. Maybe do transfer learning in 3D.

