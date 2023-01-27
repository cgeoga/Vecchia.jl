
# A few constants for options and things.
const DEF_OPTS = Dict(:max_iter=>30, :eta=>1/8, :verbose=>true,
                      :delta_init=>1.0, :delta_max=>100.0, :delta_min=>1e-8,
                      :fatol=>1e-8, :frtol=>1e-10, :gtol=>1e-5, :nstol=>1e-4)

const MOI_OK = (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, 
                MOI.ALMOST_INFEASIBLE, MOI.ALMOST_DUAL_INFEASIBLE,
                MOI.LOCALLY_SOLVED)

# Returning these as named tuples instead of structs so that you can serialize
# the objects without needing to bring in whatever package defines the struct type.
successresult(s, x, v, i, t) = (status=0, criterion=s, minimizer=x, minval=v, iter=i, tol=t)
failureresult(s, x, i, er=nothing) = (status=-1, criterion=s, minimizer=x, iter=i, error=er)

# updating the size of the trust region, including some print output information
# if the verbose flag is given.
function update_region(rho, delta, delta_max, normx, vrb)
  if rho < 0.25
    vrb && print("-")
    delta /= 4
  elseif rho > 0.75 && normx > 0.999*delta
    vrb && print("+")
    delta = min(2*delta, delta_max)
  end
  delta
end

# Checking from highest to lowest quality of convergence. At some point, this
# should really just check the KKT conditions directly, and that should be the
# only option. Or at the least, the default tols for things like the objective
# should be zero or practically zero.
function check_convergence(x, fkm1, fk, frtol, fatol, gk, gtol, hk, nstol, iter)
  ns = Symmetric(hk)\gk
  if maximum(abs, ns) < nstol
    return successresult(:NS_TOL_ACHIEVED,   x, fk, iter, nstol)
  elseif maximum(abs, gk) < gtol
    return successresult(:GRAD_TOL_ACHIEVED, x, fk, iter,  gtol)
  elseif abs(fk - fkm1) < fatol
    return successresult(:OBJ_ATOL_ACHIEVED, x, fk, iter, fatol)
  elseif abs(fk - fkm1)/min(abs(fk), abs(fkm1)) < frtol
    return successresult(:OBJ_RTOL_ACHIEVED, x, fk, iter, frtol)
  end
  nothing
end

# This uses JuMP+Ipopt to solve the local quadratic subproblem of the sqptr
# optimizer. Conventionally I think people would say that it is super overkill
# to exactly solve the problem, but GP optimization problems live in sort of a
# funny space where they are very cursed but very low-dimensional. Getting a
# Hessian once can be super expensive, and exactly solving this problem will be
# cheap by comparison. So it is my hope/expectation, which has been sort of born
# out through my own ad-hoc testing, that it can save at least a few iterations
# total to do this exactly.
function solve_qp(xk, gk, hk::Symmetric, S::Symmetric, delta, 
                  box_lower, box_upper, iter)
  try
    # Check that the TR norm matrix is positive definite, or else things might blow up:
    @assert rank(S) == size(S,1) "Please provide a full-rank matrix for the trust region norm."
    n   = length(xk)
    lb  = box_lower - xk
    ub  = isnothing(box_upper) ? nothing : box_upper - xk
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "sb", "yes") # turn off the banner
    if isnothing(box_upper)
      @variable(model, x[i=1:n] >= lb[i])
    else
      @variable(model, lb[i] <= x[i=1:n] <= ub[i])
    end
    @objective(model, Min, dot(x, gk) + dot(x, hk, x)/2)
    @constraint(model, dot(x, S, x) <= delta^2)
    optimize!(model)
    return (termination_status(model), value.(x))
  catch er
    return failureresult(:SUBPROB_ERROR, xk, iter, er)
  end
end

# This is a default that I'll probably want to change at some point. Part of why
# I'm using JuMP+Ipopt is to handle box constraints, but another part is because
# I want to be able to write more general formulations for trust regions and
# make Convex.jl handle solving those things.
id_tr_matrix(x, h) = Symmetric(Matrix{Float64}(I(length(x))))

"""
sqptr: minimize a function f with a TR formulation. At present, supports only box constraints, and is effectively just a TR code that solves the subproblem exactly using JuMP+Ipopt. But next on the list is to implement some variety of SQP to handle real constraints. I'm just undecided about which method I'd like to try. 
"""
function sqptr_optimize(f, init; 
                        fgh=AutoFwdfgh(f, length(init)),
                        box_lower=fill(-floatmax(), length(init)), 
                        box_upper=nothing, 
                        regionfunction=id_tr_matrix, 
                        kwargs...)
  # read in args, set up x0:
  n     = length(init)
  args  = merge(DEF_OPTS, Dict(kwargs))
  delta = args[:delta_init]*sqrt(n)
  (x0, xp, fxp) = (copy(init), copy(init), NaN)
  (fk, gk, hk)  = (NaN, fill(NaN, n), fill(NaN, n, n))
  vrb   = args[:verbose]
  # main loop:
  for j in 1:args[:max_iter]
    vrb && pretty_print_vec(x0)
    try
      (fk, gk, hk) = fgh(x0)
    catch er
      return failureresult(:FGH_ERROR, x0, j, er)
    end
    mk  = LocalQuadraticApprox(fk, gk, Symmetric(hk))
    s   = regionfunction(x0, hk)
    rho = 0.0
    while rho < args[:eta]
      vrb && print(".")
      (stat, step) = solve_qp(x0, gk, hk, s, delta, box_lower, box_upper, j)
      #in(stat, MOI_OK) || return failureresult(Symbol(:SUBPROB_FAIL_, stat), x0, j)
      if !in(stat, MOI_OK)
        vrb && print("f")
        delta /= 4
        continue
      end
      xp    = x0 .+ step
      fxp   = f(xp)
      rho   = (fk - fxp)/(fk - mk(step))
      delta = update_region(rho, delta, args[:delta_max]*sqrt(n), norm(step), vrb)
      delta < args[:delta_min] && return failureresult(:REGION_TOO_SMALL, x0, j)
    end
    vrb && println()
    res = check_convergence(xp, fk, fxp, args[:frtol], args[:fatol], gk, args[:gtol], 
                            hk, args[:nstol], j)
    isnothing(res) || return res
    x0 .= xp
  end
  failureresult(:FAIL_MAX_ITER, x0, args[:max_iter])
end

