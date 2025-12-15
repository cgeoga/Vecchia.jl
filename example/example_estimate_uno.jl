
using NLPModels, ForwardDiff, UnoSolver

include("example_setup.jl")

const approx = VecchiaApproximation(pts, matern, sim)

solver = NLPModelsSolver(uno; preset="filtersqp")
est = vecchia_estimate(approx, init[1:3], solver;
                       box_lower=[1e-8, 1e-8, 0.25], 
                       box_upper=[10.0, 10.0, 5.0])

