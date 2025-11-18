
using NLPModels, ForwardDiff, UnoSolver

# This just simulates some data and stuff. Assuming that you have data already
# in your application, no need to read this file.
include("example_setup.jl")

# Create the VecchiaConfig, which specifies the prediction and conditioning
# sets. This knn- and random ordering-based configuration is a generic choice
# that works well in many settings.
const cfg = knnconfig(sim, pts, 10, matern)

est = Vecchia.optimize(cfg, init[1:3], Vecchia.UnoNLPSolver();
                       box_lower=[1e-8, 1e-8, 0.25], 
                       box_upper=[10.0, 10.0, 5.0])

