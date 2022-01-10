using Ipopt, ForwardDiff
import Vecchia: ipopt_hessian

# For reference, compute the exact MLE. This file also loads example_setup.jl,
# which creates the simulated data and stuff.
println("Exact estimation for reference:")
include("example_exact_reference.jl")
println("\nExact MLE:\n")
display(exact_mle)
println("\n\nExact Hessian:\n")
display(exact_obs_inf_mat)
print("\n\n\n")

# Very important for multi-threading and small linear algebra. I don't set this
# in the package source, but you really should if you're going to use this.
BLAS.set_num_threads(1)

# Load in the functions for the objective (_nll), gradient (grad!), and Hessian
# (hess), which are made slightly more complicated to squeeze out redundant AD
# computations.
include("ad_nll_derivatives.jl")

# Set up and solve the problem in Ipopt.
prob = CreateIpoptProblem(
         2,ones(2).*1.0e-3, ones(2).*100.0,
         0,Float64[], Float64[], 0,3,
         _nll,  (args...)->nothing,
         grad!, (args...)->nothing,
         (x,r,c,o,l,v)->ipopt_hessian(x,r,c,o,l,v,hess,Function[],0))
AddIpoptNumOption(prob, "tol",      1.0e-5)
AddIpoptIntOption(prob, "max_iter", 1000)
prob.x = ones(2)
println("Solving problem with Vecchia approximation:")
@time status = IpoptSolve(prob)

# Here are your MLE and observed information matrix, with pretty display:
mle = copy(prob.x)
obs_inf_mat = hess(mle)
println("\nMLE:\n")
display(mle)
println("\n\nHessian:\n")
display(obs_inf_mat)
print("\n\n\n")
