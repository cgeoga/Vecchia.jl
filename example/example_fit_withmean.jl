
using NLPModels, ForwardDiff, UnoSolver, sequentialknn_jll

include("example_setup.jl")

meanfun(x, params)   = params[1]*norm(x - SA[0.5, 0.5])

# add a mean function to the data just to demonstrate fitting.
data_train .+= [meanfun(x, [10.0]) for x in pts_train]
data_pred  .+= [meanfun(x, [10.0]) for x in pts_pred]

# specify the approximation for the data.
const approx = VecchiaApproximation(pts_train, matern, data_train;
                                    meanfun=meanfun)

# compute the approximate mle.
solver = NLPModelsSolver(uno; preset="filtersqp")
init   = Parameters(cov_params=ones(3), mean_params=[0.0])
mle    = vecchia_estimate(approx, init, solver)

# now predict at the un-observed locations.
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

