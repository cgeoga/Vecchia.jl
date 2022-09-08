
using LinearAlgebra, StaticArrays, StableRNGs, Vecchia, BesselK

# A Matern kernel with no nugget---fully AD compatible, including with respect
# to the smoothness nu, thanks to BesselK.jl. If you use this kernel function
# and BesselK.jl, please throw the paper for BesselK.jl a citation in your paper!
function matern_nonug(x, y, params)
  (sg, rho, nu, nug) = params
  dist = norm(x-y)
  iszero(dist) && return sg*sg
  arg = sqrt(2*nu)*dist/rho
  (sg*sg*(2^(1-nu))/BesselK.gamma(nu))*BesselK.adbesselkxv(nu, arg)
end

# A generic function to simulate that gives back both the data with the noise
# and without the noise, because there are two estimation examples that use
# both.
function matern_simulate(pts, params, rng)
  S    = Symmetric([matern_nonug(x,y,params) for x in pts, y in pts])
  St   = cholesky(S)
  pure = St.L*randn(rng, length(pts)) 
  (pure, pure .+ randn(rng, length(pure)).*params[end])
end

# Define some const values for the simulation:
if !(@isdefined sim)
  const rng  = StableRNG(123)
  const tru  = [5.0, 0.05, 2.25, 0.25]
  const pts  = [SVector{2,Float64}(rand(2)...) for _ in 1:2_000]
  const saa  = rand(rng, (-1.0, 1.0), length(pts), 72)
  const init = [2.5, 0.1, 1.0, 0.5]
  const (sim, sim_nug) = matern_simulate(pts, tru, rng)
end

