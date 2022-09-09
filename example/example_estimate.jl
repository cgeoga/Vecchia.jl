
# This just simulates some data and stuff. Assuming that you have data already
# in your application, no need to read this file.
include("example_setup.jl")

# Create the VecchiaConfig, which specifies the prediction and conditioning
# sets. This package just uses a generic K-d tree for this, but there are other
# options that might work meaningfully better in at least some edge cases. 
const cfg = Vecchia.kdtreeconfig(sim, # your simulated data, a Matrix{Float64}.
                                 pts, # locations, a Vector{SVector{D,Float64}}.
                                 5,   # size of each leaf/prediction set.
                                 3,   # number of past leaves to condition on.
                                 matern) 

# Now just compute the estimator and let autodiff and Ipopt take care of the rest!
# Note that you can provide kwargs here for the optimizer. But if you're
# providing your own optimizer you're probably customizing more than that anyway.
const estimator = vecchia_estimate(cfg, init[1:3]) # config and init.

