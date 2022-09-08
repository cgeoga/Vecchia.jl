
using LinearAlgebra, StaticArrays, StableRNGs, Vecchia

# Data size:
const sz = 724

# Covariance function (simple SVector format for pts):
kfn(x,y,p) = p[1]*exp(-norm(x-y)/p[2])*(1.0+norm(x-y)/p[2])

# Covariance function (scalar format for pts to enable SIMD assembly):
function kfn_scalar(x1, x2, y1, y2, p)
  nrm = sqrt((x1-y1)^2 + (x2-y2)^2)
  p[1]*exp(-nrm/p[2])*(1+nrm/p[2])
end

# Choose some random locations to make measurements.
const rng = StableRNG(12345)
const pts = [SVector{2, Float64}(randn(rng, 2)) for _ in 1:sz]
const pts_1d  = range(0.0, 1.0, length=32)
const reg_pts = vec(map(x->SVector{2,Float64}(x...), 
                        Iterators.product(pts_1d, pts_1d)))

# Pick true parameters, get the true covariance matrix, and simulate with
# Cholesky factor.
tru_parm = [3.0, 0.5]
truK = Symmetric([kfn(x,y,tru_parm) for x in pts, y in pts])
const sim = cholesky(truK).L*randn(StableRNG(123), sz)

