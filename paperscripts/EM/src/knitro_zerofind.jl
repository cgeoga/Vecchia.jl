
# Just a workaround until the nnzJ thing can be used directly.
function dense_jac_structure(n, nconstr)
  M = vec(collect(Iterators.product(Int32.(0:(n-1)), Int32.(0:(nconstr-1)))))
  (getindex.(M,1), getindex.(M,2))
end

# Constraints get passed in here as an iterable container of the PAIRS
#
# [(lower_j, upper_j) => fn_j for j in 1:nconstr]
#
# so that, e.g., constraints[j].second(x) evaluates the constraints at value x.
function knitro_nlsolve_grad(obj, ini; kwargs...)
  # Merge in args:
  optargs = merge(DEF_KNITRO_KWARGS, Dict(kwargs))
  # Extract the constraint functions:
  nconstr  = length(ini)
  # Prepare the lagrangian function:
  lagrange = (x, l) -> begin
    _g = ForwardDiff.gradient(obj, x)
    dot(_g, l[1:length(_g)])
    end
  # Objective callback:
  callback_obj = (kc, cb, evalRequest, evalResult, userParams) -> begin
    x = evalRequest.x
    evalResult.obj[1] = 0.0
    ForwardDiff.gradient!(evalResult.c, obj, x)
    0
  end
  # Objective gradient and constraint jacobian callback:
  callback_objg_constrj = (kc, cb, evalRequest, evalResult, userParams) -> begin
    x = evalRequest.x
    evalResult.objGrad .= 0.0
    evalResult.jac .= vec(ForwardDiff.hessian(obj, x))
    0 
  end
  # Lagrangian hessian. Oddly in this case, it doesn't actually seem to help to
  # provide it versus just giving back a matrix of zeros. A matrix of zeros is
  # totally not correct, but here we are. This also appears to be better than
  # using BFGS or SR(1).
  callback_lagh = (kc, cb, evalRequest, evalResult, userParams) -> begin
    x = evalRequest.x
    s = evalRequest.sigma
    l = evalRequest.lambda
    #h = ForwardDiff.hessian(z->lagrange(z,l), x)
    #utri_linear_buffer_rowmajor!(evalResult.hess, h) 
    evalResult.hess .= 0.0
    0
  end
  # set up the problem:
  kc = KNITRO.KN_new()
  KNITRO.KN_load_param_file(kc, dirname(@__FILE__)*"/knitro_opts/nlsolve.opt")
  # add the variables:
  n  = length(ini)
  KNITRO.KN_add_vars(kc, n)
  KNITRO.KN_set_var_lobnds_all(kc, prepare_bounds(n, optargs[:box_lower]))
  KNITRO.KN_set_var_upbnds_all(kc, prepare_bounds(n, optargs[:box_upper]))
  # add the constraints:
  KNITRO.KN_add_cons(kc, nconstr)
  KNITRO.KN_set_con_lobnds_all(kc, fill(0.0, nconstr))
  KNITRO.KN_set_con_upbnds_all(kc, fill(0.0, nconstr))
  # set the init:
  KNITRO.KN_set_var_primal_init_values_all(kc, copy(ini))
  # Set the objective and constraint callback:
  cb = KNITRO.KN_add_eval_callback(kc, true, Int32.(0:(nconstr-1)), callback_obj)
  # set the gradient and constraint jacobian:
  (j_ix_1, j_ix_2) = dense_jac_structure(n, nconstr)
  KNITRO.KN_set_cb_grad(kc, cb, callback_objg_constrj,
                        jacIndexCons=j_ix_2,
                        jacIndexVars=j_ix_1)
  # set the lagrangian hessian and allow for it to be called with sigma==0.0:
  KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, callback_lagh)
  KNITRO.KN_set_param(kc, KNITRO.KN_PARAM_HESSIAN_NO_F, KNITRO.KN_HESSIAN_NO_F_ALLOW)
  # solve the problem, extract output, free KNITRO objects, and return:
  nStatus = KNITRO.KN_solve(kc)
  (status, sol, x, lam) = KNITRO.KN_get_solution(kc)
  KNITRO.KN_free(kc)
  (status=status, minimizer=x, minval=sol, error=nothing)
end

# Oddly, for our problems, it is SO much easier to make the nugget parameter too
# small and come up to it from below. YMMV, but this is quite consistent for me.
function knitro_nlsolve_grad_multi(obj, ini; 
                                   pre_shrink=0.1, shrinkfactor=0.5, 
                                   max_tries=10, kwargs...)
  _ini = copy(ini)
  _ini[end] *= pre_shrink
  for j in 1:max_tries
    succ_flag = false
    try
      res = knitro_nlsolve_grad(obj, _ini; kwargs...)
      succ_flag = (iszero(res.status) || -100<=res.status<=-199)
      succ_flag && return res
    catch er
      succ_flag = false
    end
    println("nlsove failed, shrinking nugget parameter by a factor of 2 and retrying...")
    _ini[end] *= shrinkfactor
  end
  throw(error("No successful convergence after $max_tries attempts."))
end

