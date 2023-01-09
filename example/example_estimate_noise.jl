
# Unlike in example_estimate.jl, the data here has also been polluted with
# additive noise, which sort of ruins the screening effect. So we recently wrote
# a paper (Geoga & Stein 2022 ArXiv, see the README for a URL) that uses the EM
# algorithm. Here's how to use that methodology.

# as before, this is just setup:
include("example_setup.jl")

# Just as before, let's make a config object.
#
# IMPORTANT: note that the kernel function you provide here DOES NOT add the
# nugget/measurement error variance!
const cfg = Vecchia.kdtreeconfig(sim_nug, # your simulated data, a Matrix{Float64}.
                                 pts,     # locations, a Vector{SVector{D,Float64}}.
                                 1,       # size of each leaf/prediction set.
                                 5,       # number of past leaves to condition on.
                                 matern) 

# Now let's try to estimate that, where note that the parameters now use an
# extra value that gives the VARIANCE of the measurement noise.
#
# This call also demonstrates the ways to provide extra kwargs to your optimizer
# function. "box_lower" means what you expect, giving lower box bounds for each
# of the variables.
#
const opt_kw = (:box_lower=>[0.01, 0.01, 0.25, 0.0], :verbose=>false)
const em_res = em_estimate(cfg, saa, init, optimizer_kwargs=opt_kw, 
                           warn_optimizer=false, warn_notation=false,
                           norm2tol=0, max_em_iter=10)

# Your estimator is now given as the last item in your EM path:
const em_path = em_res[3]
const em_mle  = em_path[end]

