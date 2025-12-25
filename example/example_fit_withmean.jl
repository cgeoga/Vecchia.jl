
using NLPModels, ForwardDiff, UnoSolver, sequentialknn_jll

include("example_setup.jl")

# note that this function, like the covariance function, indexes its parameters
# starting with 1. This is only okay because of the Parameters interface used
# below to provide the init to the optimizer. For a more raw indexing interface,
# see the tests.
meanfun(x, params) = params[1]*norm(x - SA[0.5, 0.5])

# add a mean function to the data just to demonstrate fitting.
data_train .+= [meanfun(x, [10.0]) for x in pts_train]
data_pred  .+= [meanfun(x, [10.0]) for x in pts_pred]

# specify the approximation for the data.
const approx = VecchiaApproximation(pts_train, matern, data_train;
                                    meanfun=meanfun)

# compute the approximate mle. Note now that init is a Parameters object, which
# allows you to write both your mean and covariance functions with one-based
# indexing. And now mle will come out as a Parameters object again.
solver = NLPModelsSolver(uno; preset="filtersqp")
init   = Parameters(cov_params=ones(3), mean_params=[0.0])
mle    = vecchia_estimate(approx, init, solver)

# now predict at the un-observed locations. Even though mle is a Parameters
# object, everything will just work as expected.
pred   = predict(approx, pts_pred, mle)

# summarize the first few results:
using Printf
@printf "\n\n** A few predictions **\n"
@printf "-----------------------\n"
@printf "True value   Prediction\n"
@printf "-----------------------\n"
for j in 1:5
  @printf "  % 01.3f      % 01.3f\n" data_pred[j] pred[j]
end

