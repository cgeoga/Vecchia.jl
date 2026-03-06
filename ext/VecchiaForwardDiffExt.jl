
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

  struct ExpectedFisherBuf
    nll::Vector{Float64}
    grad::Vector{Float64}
    fish::Matrix{Float64}
    primal_cvar_buf::Matrix{Float64}
    kwts_jac_buf::Matrix{Float64}
    primal_U_kwts_jac_buf::Matrix{Float64}
  end

  function ExpectedFisherBuf(cfg, params)
    grad       = zeros(length(params))
    fish       = zeros(length(params), length(params))
    kwts_jac   = zeros(maximum(length, cfg.condix), length(params))
    U_kwt_jac  = zeros(maximum(length, cfg.condix), length(params))
    primal_cc  = zeros(maximum(length, cfg.condix), maximum(length, cfg.condix))
    ExpectedFisherBuf([0.0], grad, fish, primal_cc, kwts_jac, U_kwt_jac)
  end


  function _efish_term(cfg::Vecchia.SingletonVecchiaApproximation{Vecchia.ZeroMean,P,F}, 
                       cbuf, ebuf, params, j::Int) where{P,F}
    # compute the conditional variance and kriging weights for each point.
    ndata = size(cfg.data, 2)
    cj    = cfg.condix[j]
    ptj   = cfg.pts[j]
    cvar  = Vecchia.prepare_conditional!(cbuf, cfg, j, params)
    cres  = view(cbuf.buf_cres, 1:length(cj))
    kwts  = view(cbuf.buf_kwts, 1:length(cj))
    nllj  = sum(1:ndata) do k
      cμ  = dot(view(cfg.data, cj, k), kwts)
      (log(cvar) + ((cfg.data[j,k] - cμ)^2)/cvar)/2
    end
    # update the primal value of the nll real quick.
    ebuf.nll[1] += ForwardDiff.value(nllj)
    # update the gradient and fisher information accordingly.
    grad_parts = ForwardDiff.partials(nllj)
    cvar_parts = ForwardDiff.partials(cvar)
    cvar_value = ForwardDiff.value(cvar)
    for k in 1:length(params)
      ebuf.grad[k] += grad_parts[k]
      for l in 1:length(params)
        ebuf.fish[k,l] += (cvar_parts[l]*cvar_parts[k]/abs2(cvar_value))/2
      end
    end
    # And if the conditioning set isn't empty, we _also_ need to add the
    # quadratic form with Kriging weight jacobian and the _primal_ of the
    # conditional variance.
    if !isempty(cj)
      # update the buffer for the Jacobian of the Kriging weights.
      kwts_jac = view(ebuf.kwts_jac_buf, 1:length(cj), 1:length(params))
      for j in 1:size(kwts_jac, 1)
        for k in 1:size(kwts_jac, 2)
          kwts_jac[j,k] = ForwardDiff.partials(kwts[j])[k]
        end
      end
      # update the buffer for the upper triangle of the _primal_ marginal
      # covariance of the prediction points.
      #
      # BUG: if I use UpperTriangular(primal_cc) to exploit that structure, the
      # first mul!(...) call below allocates 8 bytes. Not the first time upper
      # triangular muls/solves have had a trailing allocation issue for me, even
      # within just this package. LinearAlgebra.jl clearly has a problem
      # somewhere with triangular matrices.
      primal_cc = view(ebuf.primal_cvar_buf, 1:length(cj), 1:length(cj))
      fill!(primal_cc, 0.0)
      cond_var  = cbuf.buf_cc
      for k in 1:length(cj)
        for j in 1:k
          primal_cc[j,k] = ForwardDiff.value(cond_var[j,k])
        end
      end
      primal_U_kwts = view(ebuf.primal_U_kwts_jac_buf, 1:length(cj), 1:length(params))
      mul!(primal_U_kwts, primal_cc, kwts_jac)
      mul!(ebuf.fish, primal_U_kwts', primal_U_kwts, inv(cvar_value), 1.0)
    end
    nothing
  end

  # TODO (cg 2026/03/06 15:17): make an ExpectedFisherBuf and parallelize.
  function _nll_grad_fish(cfg::Vecchia.SingletonVecchiaApproximation{Vecchia.ZeroMean,P,F}, 
                          params) where{P,F}
    cbuf = Vecchia.cnllbuf(cfg, params)
    ebuf = ExpectedFisherBuf(cfg, params)
    for j in 1:length(cfg.condix) 
      _efish_term(cfg, cbuf, ebuf, params, j)
    end
    ebuf.fish .*= size(cfg.data, 2)
    (ebuf.nll[], ebuf.grad, ebuf.fish) 
  end

  function Vecchia.nll_grad_fish(cfg::R, params::Vector{Float64}) where{R}
    tag = ForwardDiff.Tag(R, Float64)
    N   = length(params)
    duals = map(1:length(params)) do j
      pj = ForwardDiff.Partials(ntuple(k->Float64(k==j), N))
      ForwardDiff.Dual{typeof(tag)}(params[j], pj)
    end
    _nll_grad_fish(cfg, duals)
  end
end

