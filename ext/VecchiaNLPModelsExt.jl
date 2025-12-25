
module VecchiaNLPModelsExt

  using Vecchia, ForwardDiff, NLPModels

  mutable struct VecchiaNLPModel{C,T,S} <: AbstractNLPModel{T,S}
    cfg::C
    meta::NLPModelMeta{T,S}
    counters::Counters
    p::Int64
  end

  default_lowerbox(init::Vector{Float64}) = zeros(length(init))
  default_upperbox(init::Vector{Float64}) = fill(Inf, length(init)) 

  function default_lowerbox(init::Parameters)
    vcat(zeros(length(init.cov_params)), fill(-Inf, length(init.mean_params)))
  end

  default_upperbox(init::Parameters) = fill(Inf, length(init))

  function Vecchia.nlp(cfg::C, init::Vector{Float64};
                       box_lower=default_lowerbox(init),
                       box_upper=default_upperbox(init)) where{C}
    meta      = NLPModelMeta(length(init); x0=init, lvar=box_lower, 
                             uvar=box_upper, hprod_available=false)
    (cov_ixs, mean_ixs) = (1:length(init), 1:length(init))
    cache_cfg = Vecchia.adcachewrapper(cfg, cov_ixs, mean_ixs)
    VecchiaNLPModel(cache_cfg, meta, Counters(), length(init))
  end

  function Vecchia.nlp(cfg::C, init::Parameters;
                       box_lower=default_lowerbox(init),
                       box_upper=default_upperbox(init)) where{C}
    _init      = vcat(init.cov_params, init.mean_params)
    meta       = NLPModelMeta(length(init); x0=collect(init), lvar=box_lower, 
                              uvar=box_upper, hprod_available=false)
    ncovparams = length(init.cov_params)
    nparams    = length(init.cov_params) + length(init.mean_params)
    cov_ixs    = 1:ncovparams
    mean_ixs   = (ncovparams+1):nparams
    cache_cfg  = Vecchia.adcachewrapper(cfg, cov_ixs, mean_ixs)
    VecchiaNLPModel(cache_cfg, meta, Counters(), length(init))
  end

  function NLPModels.obj(vnlp::VecchiaNLPModel{C,T,S}, x) where{C,T,S}
    return try
      Vecchia._primal(vnlp.cfg, x)
    catch
      NaN
    end
  end

  function NLPModels.grad!(vnlp::VecchiaNLPModel{C,T,S}, x, g) where{C,T,S}
    _g = try
      Vecchia._gradient(vnlp.cfg, x)
    catch
      fill(NaN, length(x))
    end
    g .= _g
  end

  function NLPModels.hess_structure!(vnlp::VecchiaNLPModel{C,T,S}, 
                                     hrows, hcols) where{C,T,S}
    idx = 1
    for row in 1:vnlp.p
      for col in 1:row
        hrows[idx] = row
        hcols[idx] = col
        idx       += 1
      end
    end
    (hrows, hcols)
  end

  function NLPModels.hess_coord!(vnlp::VecchiaNLPModel{C,T,S}, 
                                 x::AbstractVector{Float64}, 
                                 hvals::AbstractVector{Float64}; 
                                 obj_weight=1) where{C,T,S}
    _h = try
      Vecchia._hessian(vnlp.cfg, x)
    catch
      fill(NaN, (length(x), length(x)))
    end
    _h .*= obj_weight
    (sz, linear_idx) = (length(x), 1)
    for col_idx in 1:sz
      for row_idx in 1:col_idx
        hvals[linear_idx] = _h[row_idx, col_idx]
        linear_idx   += 1
      end
    end
    hvals
  end

  function Vecchia.optimize(obj, init, solver::Vecchia.NLPModelsSolver;
                            box_lower=fill(0.0, length(init)),
                            box_upper=fill(Inf, length(init)))
    nlp = Vecchia.nlp(obj, init, box_lower=box_lower, box_upper=box_upper)
    solver.solver(nlp; solver.opts...).solution
  end

end

