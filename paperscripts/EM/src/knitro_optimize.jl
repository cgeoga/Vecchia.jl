
const DEF_KNITRO_KWARGS = Dict(:box_lower=>[1e-4,  1e-4, 0.25, 0.0], 
                               :box_upper=>[100.0, 5.0,  4.0,  100.0], 
                               :check_derivatives=>false, 
                               :finite_diff_hessian=>false)

function ltri_linear_buffer!(lbuf, mbuf)
  linear_idx = 1
  for col_idx in 1:size(mbuf, 2)
    for row_idx in 1:col_idx
      lbuf[linear_idx] = mbuf[row_idx, col_idx]
      linear_idx += 1
    end
  end
  nothing
end

function utri_linear_buffer_rowmajor!(lbuf, mbuf)
  linear_idx = 1
  for row_idx in 1:size(mbuf, 2)
    for col_idx in row_idx:size(mbuf, 2)
      lbuf[linear_idx] = mbuf[row_idx, col_idx]
      linear_idx += 1
    end
  end
  nothing
end

function knitro_ad_callbacks(obj)
  callback_obj = (kc, cb, evalRequest, evalResult, userParams) -> begin
    x = evalRequest.x
    evalResult.obj[1] = obj(x)
    0
  end
  # gradient:
  callback_objg! = (kc, cb, evalRequest, evalResult, userParams) -> begin
    x = evalRequest.x
    ForwardDiff.gradient!(evalResult.objGrad, obj, x)
    0
  end
  # hessian:
  callback_objh! = (kc, cb, evalRequest, evalResult, userParams) -> begin
    x = evalRequest.x
    sigma = evalRequest.sigma
    scaled_hess = ForwardDiff.hessian(obj, x).*sigma
    utri_linear_buffer_rowmajor!(evalResult.hess, scaled_hess)
    0
  end
  (callback_obj, callback_objg!, callback_objh!)
end

function knitro_gpmaxlik_callbacks(kernel, pts, data, saa, nparams)
  # create the kernel derivatives:
  dkernelv = prepare_dkernelv(kernel, nparams)
  callback_obj = (kc, cb, evalRequest, evalResult, userParams) -> begin
    x = evalRequest.x
    evalResult.obj[1] = GPMaxlik.gnll(pts, data, kernel, dkernelv, x, 
                                      saa=nothing, nll=true, grad=false, 
                                      fish=false, vrb=true).nll
    0
  end
  # gradient:
  callback_objg! = (kc, cb, evalRequest, evalResult, userParams) -> begin
    x = evalRequest.x
    res = GPMaxlik.gnll(pts, data, kernel, dkernelv, x, 
                        saa=nothing, nll=false, grad=true, 
                        fish=false, vrb=true).grad
    evalResult.objGrad .= res
    0
  end
  # hessian:
  callback_objh! = (kc, cb, evalRequest, evalResult, userParams) -> begin
    x = evalRequest.x
    sigma = evalRequest.sigma
    res = GPMaxlik.gnll(pts, data, kernel, dkernelv, x, 
                        saa=saa, nll=false, grad=false, 
                        fish=true, vrb=true).fish.*sigma
    utri_linear_buffer_rowmajor!(evalResult.hess, res)
    0
  end
  (callback_obj, callback_objg!, callback_objh!)
end

function knitro_optimize(obj, ini; kwargs...)
  (callback_obj, callback_objg!, callback_objh!) = knitro_ad_callbacks(obj)
  knitro_optimize(callback_obj, callback_objg!, callback_objh!, ini; kwargs...)
end

function knitro_optimize(callback_obj, callback_objg!, callback_objh!, 
                         ini; kwargs...)
  try
    optargs = merge(DEF_KNITRO_KWARGS, Dict(kwargs))
    # set up the problem:
    n  = length(ini)
    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, dirname(@__FILE__)*"/knitro_opts/optimize.opt")
    KNITRO.KN_add_vars(kc, n)
    KNITRO.KN_set_var_lobnds_all(kc, prepare_bounds(n, optargs[:box_lower]))
    KNITRO.KN_set_var_upbnds_all(kc, prepare_bounds(n, optargs[:box_upper]))
    KNITRO.KN_set_var_primal_init_values_all(kc, copy(ini))
    cb = KNITRO.KN_add_objective_callback(kc, callback_obj)
    KNITRO.KN_set_cb_grad(kc, cb, callback_objg!)
    if !optargs[:finite_diff_hessian]
      KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, callback_objh!)
      KNITRO.KN_set_param(kc, KNITRO.KN_PARAM_HESSIAN_NO_F, KNITRO.KN_HESSIAN_NO_F_ALLOW)
    end
    # check derivatives if requested:
    if optargs[:check_derivatives]
      KNITRO.KN_set_param(kc, KNITRO.KN_PARAM_DERIVCHECK, KNITRO.KN_DERIVCHECK_ALL)
    end
    # solve the problem:
    nStatus = KNITRO.KN_solve(kc)
    (status, sol, x, lambda_) = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)
    return (status=status, minimizer=x, minval=sol, error=nothing)
  catch er
    return (status=12345, minimizer=fill(NaN, length(ini)), minval=NaN, error=er)
  end
end
