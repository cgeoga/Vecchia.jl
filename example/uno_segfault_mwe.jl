
using LinearAlgebra, StaticArrays, Vecchia, BesselK, NLPModels, ForwardDiff, UnoSolver

function matern_simulate(pts, params)
  S = Symmetric([matern(x,y,params) for x in pts, y in pts])
  cholesky!(S).U'*randn(length(pts))
end

pts  = rand(SVector{2,Float64}, 1_000)
init = [2.5, 0.1, 1.1]
sim  = matern_simulate(pts, [5.0, 0.1, 2.25])

const cfg = knnconfig(sim, pts, 10, matern)

est = Vecchia.optimize(cfg, init[1:3], Vecchia.UnoNLPSolver();
                       box_lower=[1e-8, 1e-8, 0.25], 
                       box_upper=[10.0, 10.0, 5.0])

