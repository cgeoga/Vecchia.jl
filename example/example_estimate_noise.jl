
using JuMP, Ipopt

# Unlike in example_estimate.jl, the data here has also been polluted with
# additive noise, which sort of ruins the screening effect. So we recently wrote
# a paper (Geoga & Stein 2023 JCGS, see the README for a URL) that uses the EM
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
ipopt  = optimizer_with_attributes(Ipopt.Optimizer, "tol"=>1e-3, "sb"=>"yes", "print_level"=>0)
em_res = em_estimate(cfg, saa, init, solver=ipopt, box_lower=[1e-8, 1e-8, 0.25, 0.0],
                     errormodel=Vecchia.ScaledIdentity(length(pts)),
                     warn_notation=false)

# Your estimator is now given as the last item in your EM path:
path    = em_res.path
em_mle  = path[end]

