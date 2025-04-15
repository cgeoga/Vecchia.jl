
using GALAHAD, Accessors, ForwardDiff

# This just simulates some data and stuff. Assuming that you have data already
# in your application, no need to read this file.
include("example_setup.jl")

# A perfectly fine default configuration.
const cfg = knnconfig(sim, pts, 10, matern)

# Now just compute the estimator and let autodiff and Ipopt take care of the rest!
# Note that you can provide kwargs here for the optimizer. But if you're
# providing your own optimizer you're probably customizing more than that anyway.
solver  = Vecchia.TRBSolver()
mle     = Vecchia.vecchia_estimate(cfg, init[1:3], solver; 
                                   box_lower=[1e-8, 1e-8, 0.25],
                                   box_upper=[10.0, 10.0, 5.0]) 

# Once you have fitted your model, you can predict using the same style of knn
# conditioning:
preds = knnpredict(cfg, mle, pts_hold; ncondition=80)
@show maximum(abs, preds - holdout_truth)

