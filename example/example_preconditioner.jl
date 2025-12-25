
# setup: say you have a positive definite kernel matrix and want to solve a
# linear system with a Krylov method. Here is how you can get a fast and
# effective preconditioner from a Vecchia approximation. We'll use the matern
# kernel here to illustrate.

using Krylov

include("example_setup.jl")

# here is the preconditioner. Note that we don't need to pass the data in here,
# because we won't be using it. And for a preconditioner, I recommend cranking
# up the number of conditioning points a bit past the defaults, which are tuned
# for likelihoods.
appx = VecchiaApproximation(pts, matern; conditioning=KNNConditioning(30))
pre  = rchol_preconditioner(appx, [5.0, 0.1, 2.25])

# the dense exact covariance matrix for comparison.
M = [matern(x, y, [5.0, 0.1, 2.25]) for x in pts, y in pts]

# a sample RHS.
v = collect(1.0:length(pts))

# a Vecchia preconditioner in action (try re-running this without the
# preconditioner and see how slowly it converges...)
sol2 = cg(Symmetric(M), v; M=pre, ldiv=false, verbose=1) # ~ 15 iterations

