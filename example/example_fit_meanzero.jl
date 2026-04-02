
using NLPModels, ForwardDiff, UnoSolver, sequentialknn_jll

include("example_setup.jl")

# specify the approximation for the data.
const approx = VecchiaApproximation(pts_train, matern, data_train)

# compute the approximate mle.
solver = NLPModelsSolver(uno; preset="filtersqp", TR_radius=0.1)
mle    = vecchia_estimate(approx, ones(3), solver;
                          # New in Version 0.12.8+: setting expected_fisher=true
                          # means that expected Fisher matrices will be used in
                          # place of true Hessians (a la Fisher scoring). These
                          # are faster to compute and only require first
                          # derivatives of the kernel. But they are not the
                          # exact Hessian, and so you may need to be careful in
                          # initialization (not TR_radius=0.01 in the solver
                          # options above!).
                          expected_fisher=true,
                          box_lower=[1e-8, 1e-8, 0.25], 
                          box_upper=[10.0, 10.0, 5.0])

# now predict at the un-observed locations.
preds  = predict(approx, pts_pred, mle)
cmean  = conditional_mean(preds)

# summarize the first few results:
using Printf
@printf "\n\n** A few predictions **\n"
@printf "-----------------------\n"
@printf "True value   Prediction\n"
@printf "-----------------------\n"
for j in 1:5
  @printf "  % 01.3f      % 01.3f\n" data_pred[j] cmean[j]
end

