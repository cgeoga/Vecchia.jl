
prepare_bounds(n, arg::Float64) = fill(arg, n)
prepare_bounds(n, argv::Vector{Float64}) = argv

function vecchia_sigma(cfg, arg)
  Us = sparse(Vecchia.rchol(cfg, arg, issue_warning=false))
  Symmetric(inv(Matrix(Us*Us')))
end

function vecchia_precisionU(cfg, arg)
  Us = sparse(Vecchia.rchol(cfg, arg, issue_warning=false))
  Symmetric(Us*Us')
end

function ipopt_check_code(status)
  status == 12345 ? :FAIL_USER_CODE_ERROR : Ipopt._STATUS_CODES[status]
end

function ipopt_optimize(obj, ini; kwargs...)
  try
    optargs = merge(DEF_IPOPT_KWARGS, Dict(kwargs))
    arglen  = length(ini)
    box_l   = prepare_bounds(arglen, optargs[:box_lower])
    box_u   = prepare_bounds(arglen, optargs[:box_upper])
    objg    = (p,g)->ForwardDiff.gradient!(g, obj, p)
    objh    = p -> ForwardDiff.hessian(obj, p)
    # Here's the one opaque line of this function. Ipopt wants the Hessian
    # function to give the _lagrangian_ hessian, which is the hessian of obj(p) *
    # lambda*sum_{j=1}^n g_j(p), where g_j are the constraint functions. This is a
    # bit of a mess to put together, so I just have a little helper function to do
    # it. Not super pretty, but it isn't wrong...so whatever, for now.
    ipopth  = (x,r,c,o,l,v)->ipopt_hessian(x,r,c,o,l,v,objh,Function[],0)
    prob    = CreateIpoptProblem(arglen, box_l, box_u,      # arg count and box
                                 0, Float64[], Float64[],   # constr. count and box
                                 0,                         # size of constr. jacobian
                                 div(arglen*(arglen+1), 2), # size of hessian ltri
                                 obj,                       # objective
                                 (args...)->nothing,        # constraint function
                                 objg,                      # objective gradient
                                 (args...)->nothing,        # constraint jacobian
                                 ipopth)                    # lagrangian hessian
    AddIpoptStrOption(prob, "accept_every_trial_step", optargs[:accept_every_trial_step])
    AddIpoptNumOption(prob, "alpha_red_factor",        optargs[:alpha_red_factor])
    AddIpoptStrOption(prob, "sb",          optargs[:sb])
    AddIpoptNumOption(prob, "tol",         optargs[:tol])
    AddIpoptIntOption(prob, "max_iter",    optargs[:max_iter])
    AddIpoptIntOption(prob, "print_level", optargs[:print_level])
    prob.x = copy(ini)
    status = IpoptSolve(prob)
    restart_counter = optargs[:try_restart]
    if status == -1 && restart_counter > 0
      for j in restart_counter:-1:0
        println("Hit maximum iters, stopping at parameters:")
        display(prob.x)
        println("jittering your init and trying again (max $(optargs[:try_restart]) attempts)")
        prob.x = copy(ini .+ rand(length(ini)).*0.1)
        status = IpoptSolve(prob)
        status != -1 && break
      end
    end
    return (status=status, minimizer=copy(prob.x), minval=obj(prob.x), error=nothing)
  catch er
    return (status=12345, minimizer=fill(NaN, length(ini)), minval=NaN, error=er)
  end
end

function gpmaxlik_optimize(obj, ini; kwargs...)
  try
    objgf = p -> begin
      res = DiffResults.HessianResult(p)
      ForwardDiff.hessian!(res, obj, p)
      (DiffResults.value(res), DiffResults.gradient(res), DiffResults.hessian(res))
    end
    res = GPMaxlik.trustregion(obj, objgf, ini, vrb=true, dcut=1e-14, 
                               gtol=1e-3, rtol=1e-8, dmax=10.0)
    succ = (:NEWTONSTEP_TOL_REACHED, :OBJECTIVE_RTOL_REACHED, :OBJECTIVE_ATOL_REACHED)
    if !in(res.status, succ)
      @warn "Optimized failed with status $(res.status), but not killing. \
             Proceed with caution..."
    end
    return (status=0, minimizer=res.p, minval=res.f, error=nothing)
  catch er
    return (status=12345, minimizer=fill(NaN, length(ini)), minval=NaN, error=er)
  end
end

function nlsolve_fixedpoint(F, ini; kwargs...)
  _F   = p -> F(p) - p
  objv = (buf,p)->(buf .= _F(p))
  objv_jac = (buf,p) -> FiniteDiff.finite_difference_jacobian!(buf, objv, p)
  nlsolve(objv, objv_jac, ini, xtol=1e-4, ftol=1e-4)
end

generic_nll(Sf::Cholesky, data) = 0.5*(logdet(Sf) + sum(x->x^2, Sf.U'\data))
generic_nll(R::Diagonal, data)  = 0.5*(logdet(R) + dot(data, R\data))

# TODO (cg 2022/06/02 18:13): provide seed or rngs for reproducibility.
function gauss_sim(mu, Sig, randvals=[randn(length(mu)) for _ in 1:1_000])
  L = cholesky(Sig).L
  [mu + L*x for x in randvals]
end

function joint_nll_exact(kernel_nonug, arg, pts, zv::Vector{Vector{Float64}}, y)
  Sf = cholesky!(Symmetric([kernel_nonug(x, xp, arg) for x in pts, xp in pts]))
  R  = Diagonal(fill(arg[end], length(pts)))
  logdets = logdet(R) + logdet(Sf)
  map(zv) do z
    0.5*(logdets + sum(t->t^2, Sf.U'\z) + dot(y-z, R\(y-z)))
  end
end

function joint_nll_vecc(cfg, arg, zv::Vector{Vector{Float64}})
  y = reduce(vcat, cfg.data)
  U = Vecchia.rchol(cfg, arg, issue_warning=false)
  R = Diagonal(fill(arg[end], length(first(zv))))
  logdetR = logdet(R)
  map(zv) do z
    0.5*(logdetR + dot(y-z, R\(y-z))) + Vecchia.nll(U, hcat(z))
  end
end

function exact_nll(pts, data, kernel, arg)
  nug_kernel = (x,y,p) -> kernel(x,y,p) + Float64(x==y)*p[end]
  GPMaxlik.gnll_forwarddiff(arg, pts, data, nug_kernel)
end

function exact_nll(cfg::Vecchia.AbstractVecchiaConfig, arg)
  pts = reduce(vcat, cfg.pts)
  dat = reduce(vcat, cfg.data)
  exact_nll(pts, dat, cfg.kernel, arg)
end

function exact_vecchia_nll(cfg, arg)
  pts   = reduce(vcat, cfg.pts)
  dat   = reduce(vcat, cfg.data)
  S     = Vecchia.precisionmatrix(cfg, arg, issue_warning=false)
  #U     = sparse(Vecchia.rchol(cfg, arg, issue_warning=false))
  #Sig   = inv(Matrix(U*U'))
  Sig   = inv(Matrix(S))
  Sig .+= Diagonal(fill(arg[end], length(pts)))
  generic_nll(cholesky!(Symmetric(Sig)), dat) 
end

# Uses the identiy Sig+R = R*(inv(Sig) + inv(R))*Sig, so
# inv(Sig+R) = inv(Sig)*inv(inv(Sig)+inv(R))*inv(R).
function exact_vecchia_nll_precform(cfg, arg)
  pts  = reduce(vcat, cfg.pts)
  dat  = reduce(vcat, cfg.data)
  iS   = Symmetric(Matrix(Vecchia.precisionmatrix(cfg, arg, issue_warning=false)))
  iR   = Diagonal(fill(inv(arg[end]), length(pts)))
  iSRf = cholesky!(Symmetric(iS+iR))
  iSf  = cholesky(iS)
  logdet_term = logdet(iSRf) - logdet(iSf) - logdet(iR)
  solve_term  = dot(dat, iS*(iSRf\(iR*dat)))
  0.5*(logdet_term + solve_term)
end

function exact_vecchia_nll_noad(cfg, arg)
  dat  = reduce(vcat, cfg.data)
  S    = Vecchia.precisionmatrix(cfg, arg, issue_warning=false)
  iR   = Diagonal(fill(inv(arg[end]), length(dat)))
  SiR  = Symmetric(S+iR)
  Sf   = cholesky(S)
  SiRf = cholesky(SiR)
  zhat = SiRf\(iR*dat)
  numer = 0.5*(-logdet(Sf) - logdet(iR) + dot(dat, iR*dat))
  denom = 0.5*(-logdet(SiRf) + dot(zhat, SiR*zhat))
  numer - denom
end

function exact_vecchia_mle(cfg, init; kwargs...)
  knitro_optimize(p->exact_vecchia_nll(cfg, p); kwargs...)
end

function exact_mle(cfg, init; kwargs...)
  pts   = reduce(vcat, cfg.pts)
  dat   = reduce(vcat, cfg.data)
  nug_kernel = (x,y,p) -> cfg.kernel(x,y,p) + Float64(x==y)*p[end]
  knitro_optimize(p->GPMaxlik.gnll_forwarddiff(p, pts, dat, nug_kernel), 
                 init; kwargs...)
end

function exact_mle(pts, data, kernel, init; kwargs...)
  @info "This assumes that the kernel has the nugget in it..."
  knitro_optimize(p->exact_nll(pts, data, kernel, p), init; kwargs...)
end

function exact_mle_efish(pts, data, kernel, init, saa; kwargs...)
  @info "This assumes that the kernel has the nugget in it..."
  (call_obj, call_g!, call_h!) = knitro_gpmaxlik_callbacks(kernel, pts, data, 
                                                           saa, length(init))
  knitro_optimize(call_obj, call_g!, call_h!, init; kwargs...)
end

function vecchia_mle_withnug(cfg, init; kwargs...)
  nug_cfg = Vecchia.VecchiaConfig(cfg.chunksize, cfg.blockrank,
                                  (x,y,p)->cfg.kernel(x,y,p)+Float64(x==y)*p[end],
                                  cfg.data, cfg.pts, cfg.condix)
  knitro_optimize(p->Vecchia.nll(nug_cfg, p), init; kwargs...)
end

function check_nuggetvecchia_result(mle_withnugget)
  if !in(mle_withnugget.status, (0,1)) || !isnothing(mle_withnugget.error)
    @warn "Optimization of the Vecchia likelihood with the nugget and no \
    compensation, used to generate the initialization for the EM \
    iteration, failed. Will still provide the result to the next \
    step, but particular attention to the output is advised."
    return false
  end
  return true
end

# So, note that fxpath will automatically have my next straight Picard step.
function anderson_step(xpath, fxpath, m)
  xlen = length(first(xpath))
  @assert m <= xlen "Your m can be at most the dimension of the problem."
  # for now, a bunch of ugly copies:
  gpath  = fxpath .- xpath
  (x, X) = (xpath[end], reduce(hcat, diff(xpath))[:,max(1,end-m+1):end])
  (g, G) = (gpath[end], reduce(hcat, diff(gpath))[:,max(1,end-m+1):end])
  gamma  = factorize(G)\g
  x + g - X*gamma - G*gamma
end

@generated function splice(x, newval, idx::Val{J}, len::Val{N}) where{J,N}
  quote
    @SVector [$([:(x[$j]) for j in 1:(J-1)]...), newval, $([:(x[$j]) for j in (J+1):N]...)]
  end
end

function prepare_dkernelv(kernel, nparams)
  npv = Val(nparams)
  ntuple(nparams) do j
    vj = Val(j)
    (x,y,p)->ForwardDiff.derivative(pj->kernel(x, y, splice(p, pj, vj, npv)), p[j])
  end
end

function gpmaxlik_callbacks(kernel, pts, data, saa, nparams)
  # create the kernel derivatives:
  dkernelv = prepare_dkernelv(kernel, nparams)
  callback_obj = x -> begin
    GPMaxlik.gnll(pts, data, kernel, dkernelv, x, 
                  saa=saa, nll=true, grad=false, 
                  fish=false, vrb=true).nll
  end
  # obj, gradient, and Hessian:
  callback_objh = x -> begin
    res = GPMaxlik.gnll(pts, data, kernel, dkernelv, x, 
                        saa=saa, nll=true, grad=true, 
                        fish=true, vrb=true)
    (res.nll, res.grad, res.fish)
  end
  (callback_obj, callback_objgh)
end

function gpmaxlik_optimize_fisher(kernel, pts, data, ini, saa; kwargs...)
  try
    nparams = length(ini)
    (obj, objgf) = gpmaxlik_callbacks(kernel, pts, data, saa, nparams)
    res = GPMaxlik.trustregion(obj, objgf, ini, vrb=true, dcut=1e-14, 
                               gtol=1e-3, rtol=1e-8, dmax=10.0)
    succ = (:NEWTONSTEP_TOL_REACHED, :OBJECTIVE_RTOL_REACHED, :OBJECTIVE_ATOL_REACHED)
    if !in(res.status, succ)
      @warn "Optimized failed with status $(res.status), but not killing. \
             Proceed with caution..."
    end
    return (status=0, minimizer=res.p, minval=res.f, error=nothing)
  catch er
    println("fisher optimization failed with error $er")
    return (status=12345, minimizer=fill(NaN, length(ini)), minval=NaN, error=er)
  end
end

function ipopt_nlsolve_grad(obj, ini; kwargs...)
  # bring in some default arguments:
  optargs = merge(DEF_IPOPT_KWARGS, Dict(kwargs))
  # get the dimension of the problem:
  arglen  = length(ini)
  nconstr = arglen
  # set up box constraints for the args and, if applicable, the constraints:
  box_l = prepare_bounds(arglen, optargs[:box_lower])
  box_u = prepare_bounds(arglen, optargs[:box_upper])
  con_l = zeros(length(ini))
  con_u = zeros(length(ini))
  # function for the constraints:
  constr_eval = (p,store) -> begin 
    ForwardDiff.gradient!(store, obj, p)
  end
  # objective gradient and constraint jacobian:
  objg = (p,g)->(g.=zeros(length(p)))
  constr_eval_rev = (store, p) -> constr_eval(p, store)
  constr_jac = (p,r,c,v) -> begin 
    isnothing(v) && return jac_structure!(r, c, nconstr, length(p)) 
    h = ForwardDiff.hessian(obj, p)
    v .= vec(h') 
  end
  # Lagrangian hessian:
  lagrange_h = (p,r,c,o,l,v) -> begin 
    isnothing(v) && return hes_structure!(r,c,length(p)) 
    fill!(v, 0.0)
    nothing
  end
  # Create the final problem:
  prob = CreateIpoptProblem(arglen,  box_l, box_u,     # arg count and box
                            nconstr, con_l, con_u,     # constr. count and box
                            arglen^2,                  # size of constr. jacobian
                            div(arglen*(arglen+1), 2), # size of hessian ltri
                            _p->0.0,                   # objective
                            constr_eval,               # constraint function
                            objg,                      # objective gradient
                            constr_jac,                # constraint jacobian
                            lagrange_h)                # lagrangian hessian
  # Add all the options:
  for (opt, val) in optargs
    in(opt, (:box_lower, :box_upper, :try_restart)) && continue
    ipopt_addoption!(prob, opt, val)
  end
  # Solve and return:
  prob.x = copy(ini)
  status = IpoptSolve(prob)
  (status=status, minimizer=copy(prob.x), minval=obj(prob.x))
end

# the structure function for the hessian.
function hes_structure!(rows, cols, len)
  idx = 1
  for row in 1:len
    for col in 1:row
      rows[idx] = row
      cols[idx] = col
      idx += 1
    end
  end
  nothing
end

# Note that ipopt wants the TRANSPOSE of the constraint Jacobian, so the j and k
# in these loops will look wrong. This function could probably be written more
# simply than this, but I think the readability here is worth it, and all the
# utility functions here are ugly anyway.
function jac_structure!(rows, cols, nargs, nconstr)
  linear_idx = 1
  for j in 1:nargs
    for k in 1:nconstr
      rows[linear_idx] = j
      cols[linear_idx] = k
      linear_idx += 1
    end
  end
  nothing
end
# Could also use dispatch for this, but not sure it would be faster? It would
# look nicer than this though.
function ipopt_addoption!(prob, opt, val)
  if val isa Symbol
    AddIpoptStrOption(prob, string(opt), string(val))
  elseif val isa AbstractString
    AddIpoptStrOption(prob, string(opt), string(val))
  elseif isinteger(val)
    AddIpoptIntOption(prob, string(opt), Int64(val))
  else
    AddIpoptNumOption(prob, string(opt), val)
  end
  nothing
end

# Oddly, for our problems, it is SO much easier to make the nugget parameter too
# small and come up to it from below. YMMV, but this is quite consistent for me.
function ipopt_nlsolve_grad_multi(obj, ini; 
                                  pre_shrink=0.1, shrinkfactor=0.5, 
                                  max_tries=10, kwargs...)
  _ini = copy(ini)
  _ini[end] *= pre_shrink
  for j in 1:max_tries
    succ_flag = false
    try
      res = ipopt_nlsolve_grad(obj, _ini; kwargs...)
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
