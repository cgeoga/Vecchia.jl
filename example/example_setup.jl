
using LinearAlgebra, StaticArrays, StableRNGs, Vecchia, BesselK

#=
function matern_simulate(pts, params, rng)
  S    = Symmetric([matern(x,y,params) for x in pts, y in pts])
  St   = cholesky(S)
  St.L*randn(rng, length(pts)) 
end
=#

pts = rand(StableRNG(1234), SVector{2,Float64}, 15_000)
#sim = matern_simulate(pts, [5.0, 0.1, 2.25], StableRNG(12345))
sim = randn(15_000)

#(pts_train,  data_train)  = (pts[1:3500], sim[1:3500])
#(pts_pred, data_pred)     = (pts[3501:end], sim[3501:end])

