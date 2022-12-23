
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
# This call also demonstrates the ways to provide extra kwargs to your
# optimizer function:
#
# 1. "box_lower" means what you expect, giving lower box bounds for each of the
#    variables.
# 2. "alpha_red_factor" gives the multiplicative step size reduction in the
#    backtracing line search. In my experience Ipopt, and all line search-based
#    routines, can sometimes get stuck on problems like these, and so having the
#    line search shrink the step can be practically useful.
# 3. "tol" gives the convergence tolerance for each M step. This is of course
#    very low, but remember that in the EM algorithm you don't actually _need_
#    to optimize to high precision. Any improvement in the E function will
#    improve the likelihood of your observed data at least as much, and I've
#    found that setting this tol low can in practice help you get un-stuck
#    faster.
#
# As a closing note on this, in the paper I used KNITRO, a non-free software
# that is expensive and not casual to use, because I have found that
# interior-point + trust region strategies are by far the most effective for
# this problem, and probably for GP maximum likelihood stuff in general. I have
# a toy trust region routine in a separate package (GPMaxlik.jl), but I haven't
# even implemented box constraints in it, so it's a long way from being ready to
# compete with Ipopt. Hopefully that day will come at some point, though. 
const opt_kw = (:box_lower=>[0.01, 0.01, 0.25, 0.0], 
                :alpha_red_factor=>0.15, :tol=>1e-3)
const em_res = em_estimate(cfg, saa, init, optimizer_kwargs=opt_kw, 
                           warn_optimizer=false, warn_notation=false)

# Your estimator is now given as the last item in your EM path:
const em_path = em_res[3]
const em_mle  = em_path[end]

