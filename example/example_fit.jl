
using NLPModels, ForwardDiff, UnoSolver

include("example_setup.jl")

# specify the approximation for the data.
const approx = VecchiaApproximation(pts_train, matern, data_train)

# compute the approximate mle.
solver = NLPModelsSolver(uno; preset="filtersqp")
mle    = vecchia_estimate(approx, ones(3), solver;
                          box_lower=[1e-8, 1e-8, 0.25], 
                          box_upper=[10.0, 10.0, 5.0])

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

