
using Ipopt, ForwardDiff
import Vecchia: ipopt_hessian

include("example_setup.jl")

BLAS.set_num_threads(6)

function allocating_negloglik(kfun, params, pts, vals)
  K = cholesky!(Symmetric([kfun(x,y,params) for x in pts, y in pts]))
  0.5*logdet(K) + 0.5*dot(vals, K\vals)
end

# nll and derivatives:
_enll(p) = allocating_negloglik(kfn, p, pts, sim)
egrad!(p, store) = ForwardDiff.gradient!(store, _enll, p)
ehess(p) = ForwardDiff.hessian(_enll, p)

# Set up and solve the problem in Ipopt:
exact_prob = createProblem(2,ones(2).*1.0e-3, ones(2).*100.0,
                           0,Float64[], Float64[],
                           0,3,
                           _enll,
                           (args...)->nothing,
                           egrad!,
                           (args...)->nothing,
                           (x,m,r,c,o,l,v)->ipopt_hessian(x,m,r,c,o,l,v,
                                                          ehess,Function[],0))
addOption(exact_prob, "tol",      1.0e-5)
addOption(exact_prob, "max_iter", 1000)
exact_prob.x = ones(2)
@time estatus = solveProblem(exact_prob)
@assert iszero(estatus) "Optimization for exact likelihood problem failed."

# Here are your MLE and observed information matrix:
exact_mle = copy(exact_prob.x)
exact_obs_inf_mat = ehess(exact_mle)

