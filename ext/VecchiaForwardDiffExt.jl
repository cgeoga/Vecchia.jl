
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

end

