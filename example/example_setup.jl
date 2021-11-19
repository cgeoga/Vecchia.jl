
using LinearAlgebra, StaticArrays, StableRNGs, Vecchia

# Data size:
const sz = 724

# Covariance function.
kfn(x,y,p) = p[1]*exp(-norm(x-y)/p[2])*(1.0+norm(x-y)/p[2])

# Choose some random locations to make measurements.
const rng = StableRNG(12345)
const pts = [SVector{2, Float64}(randn(rng, 2)) for _ in 1:sz]

# Pick true parameters, get the true covariance matrix, and simulate with
# Cholesky factor.
tru_parm = [3.0, 0.5]
truK = Symmetric([kfn(x,y,tru_parm) for x in pts, y in pts])
const sim = cholesky(truK).L*randn(StableRNG(123), sz)

# Create a VecchiaConfig object. You are very much welcome to create your own
# struct that is <:Vecchia.VecchiaConfig to do something more thoughtful for
# your specific problem. The kdtree method included here is very general-use.
# This object uses "chunks" of size 64, and the 3 nearest "chunk" neighbors.
const vecc     = Vecchia.kdtreeconfig(sim, pts, 64, 3, kfn)
const nys_vecc = Vecchia.nystrom_kdtreeconfig(sim, pts, 64, 3, kfn, 72)

