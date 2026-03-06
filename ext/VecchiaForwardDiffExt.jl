
module VecchiaForwardDiffExt

  using LinearAlgebra, Vecchia, ForwardDiff

  struct EvaluationResult
    primal::Float64
    gradient::Union{Nothing, Vector{Float64}}
    hessian::Union{Nothing, Symmetric{Float64, Matrix{Float64}}}
  end 

  struct CachingADWrapper{F,G,H,R}
    fn::F
    grad_config::G
    hess_config::H
    hess_result::R
    cache::Dict{Vector{Float64}, EvaluationResult}
    cov_ixs::UnitRange{Int64}
    mean_ixs::UnitRange{Int64}
  end

  function Vecchia.adcachewrapper(fn::F, cov_ixs, mean_ixs) where{F}
    npar  = max(maximum(cov_ixs), maximum(mean_ixs))
    chunk = ForwardDiff.Chunk(zeros(npar))
    obj   = t->fn(t; cov_param_ixs=cov_ixs, mean_param_ixs=mean_ixs)
    gcfg  = ForwardDiff.GradientConfig(obj, zeros(npar), chunk)
    hres  = DiffResults.HessianResult(zeros(npar))
    hcfg  = ForwardDiff.HessianConfig(obj, hres, zeros(npar), chunk)
    cache = Dict{Vector{Float64}, EvaluationResult}()
    CachingADWrapper(obj, gcfg, hcfg, hres, cache, cov_ixs, mean_ixs)
  end

  function Vecchia._primal(cw::CachingADWrapper{F,G,H,R}, x) where{F,G,H,R}
    haskey(cw.cache, x) && return cw.cache[x].primal
    primal = cw.fn(x)
    cw.cache[x] = EvaluationResult(primal, nothing, nothing)
    primal
  end

  function Vecchia._gradient(cw::CachingADWrapper{F,G,H,R}, x) where{F,G,H,R}
    if haskey(cw.cache, x)
      x_res = cw.cache[x]
      isnothing(x_res.gradient) || return x_res.gradient
    end
    res = DiffResults.GradientResult(x)
    ForwardDiff.gradient!(res, cw.fn, x, cw.grad_config)
    store = EvaluationResult(DiffResults.value(res), 
                             DiffResults.gradient(res), 
                             nothing)
    cw.cache[x] = store
    DiffResults.gradient(res)
  end

  function Vecchia._hessian(cw::CachingADWrapper{F,G,H,R}, x) where{F,G,H,R}
    if haskey(cw.cache, x)
      x_res = cw.cache[x]
      isnothing(x_res.hessian) || return x_res.hessian
    end
    res = cw.hess_result
    ForwardDiff.hessian!(res, cw.fn, x, cw.hess_config)
    store = EvaluationResult(DiffResults.value(res), 
                             DiffResults.gradient(res), 
                             Symmetric(DiffResults.hessian(res)))
    cw.cache[x] = store
    DiffResults.hessian(res)
  end

  # TODO (cg 2026/03/06 15:17): Make non-allocating.
  function _efish_term(cfg::Vecchia.SingletonVecchiaApproximation{Vecchia.ZeroMean,P,F}, 
                       cbuf, grad, fish, params, j::Int) where{P,F}
    # compute the conditional variance and kriging weights for each point.
    cj   = cfg.condix[j]
    ptj  = cfg.pts[j]
    cvar = Vecchia.prepare_conditional!(cbuf, cfg, j, params)
    cres = view(cbuf.buf_cres, 1:length(cj))
    kwts = view(cbuf.buf_kwts, 1:length(cj))
    cμ   = dot(view(cfg.data, cj, 1), kwts)
    nllj = (log(cvar) + ((cfg.data[j,1] - cμ)^2)/cvar)/2
    # update the gradient and fisher information accordingly.
    grad_parts = ForwardDiff.partials(nllj)
    cvar_parts = ForwardDiff.partials(cvar)
    cvar_value = ForwardDiff.value(cvar)
    for k in 1:length(params)
      grad[k] += grad_parts[k]
      for l in 1:length(params)
        fish[k,l] += (cvar_parts[l]*cvar_parts[k]/abs2(cvar_value))/2
      end
    end
    # And if the conditioning set isn't empty, we _also_ need to add the
    # quadratic form with Kriging weight jacobian and the _primal_ of the
    # conditional variance.
    if !isempty(cj)
      cond_var = view(cbuf.buf_cc, 1:length(cj), 1:length(cj))
      primal_cond_var_U = UpperTriangular(ForwardDiff.value.(cond_var))
      kwts_jac = [ForwardDiff.partials(kwts[j])[k] for j in 1:length(cj), k in 1:length(params)]
      tmp      = primal_cond_var_U*kwts_jac
      fish   .+= (tmp'*tmp)./cvar_value
    end
    nothing
  end

  # TODO (cg 2026/03/06 15:17): make an ExpectedFisherBuf and parallelize.
  function _efish(cfg::Vecchia.SingletonVecchiaApproximation{Vecchia.ZeroMean,P,F}, 
                  params) where{P,F}
    size(cfg.data, 2) == 1 || throw(error("Only for m=1 right now, sorry."))
    fish  = zeros(length(params), length(params))
    grad  = zeros(length(params))
    cbuf  = Vecchia.cnllbuf(cfg, params)
    for j in 1:length(cfg.condix) 
      _efish_term(cfg, cbuf, grad, fish, params, j)
    end
    (grad, fish) 
  end

  function Vecchia.efish(cfg::R, params::Vector{Float64}) where{R}
    tag = ForwardDiff.Tag(R, Float64)
    N   = length(params)
    duals = map(1:length(params)) do j
      pj = ForwardDiff.Partials(ntuple(k->Float64(k==j), N))
      ForwardDiff.Dual{typeof(tag)}(params[j], pj)
    end
    _efish(cfg, duals)
  end
end

