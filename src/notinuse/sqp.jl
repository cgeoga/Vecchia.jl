
# A few constants for options and things.
const DEF_OPTS = Dict(:max_iter=>30, :eta=>1/8, :verbose=>true,
                      :delta_init=>1.0, :delta_max=>100.0, :delta_min=>1e-8,
                      :fatol=>1e-8, :frtol=>1e-10, :gtol=>1e-5, :nstol=>1e-4)

const MOI_OK = (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, 
                MOI.ALMOST_INFEASIBLE, MOI.ALMOST_DUAL_INFEASIBLE,
                MOI.LOCALLY_SOLVED)

struct AutoFwdfgh{F,R}
  f::F 
  res::R
end

function AutoFwdfgh(f, n::Int64)
  res = DiffResults.HessianResult(zeros(n))
  AutoFwdfgh(f, res)
end

function (f::AutoFwdfgh{F,R})(x) where{F,R}
  ForwardDiff.hessian!(f.res, f.f, x)
  (DiffResults.value(f.res), DiffResults.gradient(f.res), 
   Hermitian(DiffResults.hessian(f.res)))
end

struct AutoFwdBFGS{F,R}
  f::F 
  res::R
  xm1::Vector{Float64}
  gm1::Vector{Float64}
  g::Vector{Float64}
  Bm1::Matrix{Float64}
  B::Matrix{Float64}
end

function AutoFwdBFGS(f, n::Int64)
  AutoFwdBFGS(f, DiffResults.GradientResult(zeros(n)), zeros(n), zeros(n), 
              zeros(n), zeros(n,n), Matrix{Float64}(I(n)))
end

function (f::AutoFwdBFGS{F,R})(x) where{F,R}
  # Move the "current" gradient and Hessian approx to the old spots:
  f.gm1 .= f.g
  f.Bm1 .= f.B
  # get the new gradient, and put it in place:
  ForwardDiff.gradient!(f.res, f.f, x)
  f.g .= DiffResults.gradient(f.res)
  # now compute the new updated Hessian:
  yk = f.g - f.gm1
  sk = x   - f.xm1
  bs = f.Bm1*sk
  # TODO (cg 2022/12/23 11:26): should re-write this thoughtful to use mul!, etc.
  f.B .= f.Bm1 + (yk*yk')./dot(yk, sk) - (bs*bs')./dot(sk, f.Bm1, sk)
  # update the xm1:
  f.xm1 .= x
  # return everything:
  (DiffResults.value(f.res), f.g, Hermitian(f.B))
end

# Writing a local quadratic approximation struct to avoid creating a closure.
struct LocalQuadraticApprox
  fk::Float64
  gk::Vector{Float64}
  hk::Hermitian{Float64, Matrix{Float64}}
end
(m::LocalQuadraticApprox)(p) = m.fk + dot(m.gk, p) + dot(p, m.hk, p)/2

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
function solve_qp(xk, gk, hk::Hermitian, delta, 
                  box_lower, box_upper, iter)
  try
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
    @constraint(model, dot(x, x) <= delta^2)
    optimize!(model)
    return (termination_status(model), value.(x))
  catch er
    return failureresult(:SUBPROB_ERROR, xk, iter, er)
  end
end

"""
sqptr: minimize a function f with a TR formulation. At present, supports only box constraints, and is effectively just a TR code that solves the subproblem exactly using JuMP+Ipopt. But next on the list is to implement some variety of SQP to handle real constraints. I'm just undecided about which method I'd like to try. 
"""
function sqptr_optimize(f, init; 
                        fgh=AutoFwdfgh(f, length(init)),
                        box_lower=fill(-floatmax(), length(init)), 
                        box_upper=nothing, 
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
    mk  = LocalQuadraticApprox(fk, gk, Hermitian(hk))
    rho = 0.0
    while rho < args[:eta]
      vrb && print(".")
      (stat, step) = solve_qp(x0, gk, hk, delta, box_lower, box_upper, j)
      if !in(stat, MOI_OK) && delta > args[:delta_min]
        vrb && print("f($(stat))")
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

