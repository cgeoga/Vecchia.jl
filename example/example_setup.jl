
using LinearAlgebra, StaticArrays, StableRNGs, Vecchia, BesselK

# This is very important for getting the best performance if you use multiple
# threads for the likelihood computation (which I would highly recommend you
# do). Julia's thread scheduler and the OpenBLAS scheduler CANNOT "talk" to each
# other and will get in each other's way, so you have to choose who to give the
# threads to. For such small matrices anyway, this is the obviously better
# choice. Particularly since assembling the kernel matrices is probably the
# bottleneck in most code.
BLAS.set_num_threads(1)

# A generic function to simulate that gives back both the data with the noise
# and without the noise, because there are two estimation examples that use
# both. The function matern is courtesy of BesselK.jl.
function matern_simulate(pts, params, rng)
  S    = Symmetric([matern(x,y,params) for x in pts, y in pts])
  St   = cholesky(S)
  pure = St.L*randn(rng, length(pts)) 
  (pure, pure .+ randn(rng, length(pure)).*params[end])
end

# Define some const values for the simulation:
if !(@isdefined sim)
  const rng  = StableRNG(123)
  const tru  = [5.0, 0.05, 2.25, 0.25]
  const pts  = [SVector{2,Float64}(rand(rng, 2)...) for _ in 1:5_000]
  const saa  = rand(rng, (-1.0, 1.0), length(pts), 72)
  const init = [2.5, 0.1, 1.0, 0.5]
  const (sim, sim_nug) = matern_simulate(pts, tru, rng)
end

