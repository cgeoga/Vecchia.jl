
# This doesn't really help as of now, but I'll leave the file because hopefully
# I can at some point do something useful with this idea.

using LinearAlgebra, StaticArrays, Vecchia, ForwardDiff

exp12_kernel(x, y, p) = p[1]*exp(-norm(x-y)/p[2])

# pre-compile for dimensions 1, 2, and 3:
for d in (1, 2, 3)
  # set up configurations:
  tru   = [5.0, 0.05, 0.25]
  pts   = rand(SVector{d,Float64}, 100)
  cfg   = Vecchia.kdtreeconfig(randn(length(pts)), pts, 1, 5, exp12_kernel) 
  ncfg  = Vecchia.kdtreeconfig(randn(length(pts)), pts, 1, 5, 
                              Vecchia.NuggetKernel(exp12_kernel))
  wrap  = Vecchia.WrappedLogLikelihood(cfg)
  nwrap = Vecchia.WrappedLogLikelihood(ncfg)
  # precompile likelihood, gradient, and Hessian:
  wrap(tru)
  ForwardDiff.gradient(wrap, tru)
  ForwardDiff.hessian(wrap, tru)
  nwrap(tru)
  ForwardDiff.gradient(nwrap, tru)
  ForwardDiff.hessian(nwrap, tru)
  # precompile rCholesky:
  Vecchia.rchol(cfg, tru)
end

