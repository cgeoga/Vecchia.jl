
using NLPModels, ForwardDiff, UnoSolver
using sequentialknn_jll # an optional extension for blazing fast conditioning set design

# this script just simulates some data (pts_train, data_train, pts_pred,
# data_pred). In a real application, you would naturally bring your own
# locations and data.
include("example_setup.jl")

# specify the approximation for the data. This uses the Mat\'ern covariance
# function that looks like 
#
# matern(location_1, location_2, params)
#
# provided by BesselK.jl.
approx = VecchiaApproximation(pts_train, matern, data_train)

# compute the approximate mle that identifies the "best" kernel parameters.
solver = NLPModelsSolver(uno; preset="filtersqp", TR_radius=0.1)
mle    = vecchia_estimate(approx, ones(3), solver;
                          expected_fisher=true, # a faster Hessian proxy
                          box_lower=[1e-8, 1e-8, 0.25], 
                          box_upper=[10.0, 10.0, 5.0])

# now predict at the un-observed locations.
preds  = predict(approx, pts_pred, mle)
cmean  = conditional_mean(preds)
cvars  = conditional_variances(preds)

# summarize the first few results:
using Printf
@printf "\n\n**** A few predictions ****\n"
@printf "---------------------------\n"
@printf "True value      Prediction\n"
@printf "---------------------------\n"
for j in 1:5
  @printf "  % 01.3f      % 01.3f ± %1.2f \n" data_pred[j] cmean[j] sqrt(cvars[j])*1.96
end

