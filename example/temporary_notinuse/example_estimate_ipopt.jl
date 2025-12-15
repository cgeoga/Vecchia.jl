
using ForwardDiff, NLPModels, NLPModelsIpopt

# This just simulates some data and stuff. Assuming that you have data already
# in your application, no need to read this file.
include("example_setup.jl")

# Create the VecchiaConfig, which specifies the prediction and conditioning
# sets. This knn- and random ordering-based configuration is a generic choice
# that works well in many settings.
const cfg = Vecchia.knnconfig(sim, pts, 10, matern)

# Now just compute the estimator and let autodiff and Ipopt take care of the rest!
# Note that you can provide kwargs here for the optimizer. But if you're
# providing your own optimizer you're probably customizing more than that anyway.
solver    = NLPModelsSolver(ipopt; tol=1e-4)
estimator = vecchia_estimate(cfg, init[1:3], solver) 

