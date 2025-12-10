
module VecchiaNLPModelsExt

  using Vecchia, ForwardDiff, NLPModels

  import Vecchia.UnoNLPSolver

  mutable struct VecchiaNLPModel{C,T,S} <: AbstractNLPModel{T,S}
    cfg::C
    meta::NLPModelMeta{T,S}
    counters::Counters
    p::Int64
  end

  function Vecchia.nlp(cfg::C, init;
                       box_lower=fill(0.0, length(init)),
                       box_upper=fill(Inf, length(init))) where{C}
    meta      = NLPModelMeta(length(init); x0=init, lvar=box_lower, uvar=box_upper)
    cache_cfg = Vecchia.adcachewrapper(cfg)
    VecchiaNLPModel(cache_cfg, meta, Counters(), length(init))
  end

  function NLPModels.obj(vnlp::VecchiaNLPModel{C,T,S}, x) where{C,T,S}
    return try
      #vnlp.cfg(x)
      Vecchia._primal(vnlp.cfg, x)
    catch
      NaN
    end
  end

  function NLPModels.grad!(vnlp::VecchiaNLPModel{C,T,S}, x, g) where{C,T,S}
    _g = try
      #ForwardDiff.gradient(vnlp.cfg, x)
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
      #ForwardDiff.hessian(vnlp.cfg, x)
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

  function Vecchia.optimize(obj, init, solver::Vecchia.NLPModelsSolver, 
                            box_lower=fill(0.0, length(init)),
                            box_upper=fill(Inf, length(init)))
    nlp = Vecchia.nlp(obj, init, box_lower=box_lower, box_upper=box_upper)
    solver.solver(nlp; solver.opts...).solution
  end

end

