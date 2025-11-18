
module VecchiaUnoSolverExt

  using Vecchia, ForwardDiff, UnoSolver, NLPModels

  import Vecchia.UnoNLPSolver

  mutable struct VecchiaNLPModel{C,T,S} <: AbstractNLPModel{T,S}
    cfg::C
    meta::NLPModelMeta{T,S}
    counters::Counters
    p::Int64
  end

  function Vecchia.nlp(cfg::C, init, box_lower::Vector{Float64}, 
                       box_upper::Vector{Float64}) where{C}
    meta = NLPModelMeta(length(init); x0=init, lvar=box_lower, uvar=box_upper)
    VecchiaNLPModel(cfg, meta, Counters(), length(init))
  end

  function NLPModels.obj(vnlp::VecchiaNLPModel{C,T,S}, x) where{C,T,S}
    return try
      vnlp.cfg(x)
    catch
      NaN
    end
  end

  function NLPModels.grad!(vnlp::VecchiaNLPModel{C,T,S}, x, g) where{C,T,S}
    _g = try
      ForwardDiff.gradient(vnlp.cfg, x)
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
      ForwardDiff.hessian(vnlp.cfg, x)
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

  function Vecchia.optimize(obj::C, init, solver::Vecchia.UnoNLPSolver;
                            box_lower=fill(1e-5, length(init)),
                            box_upper=fill(100.0,length(init))) where{C}
    nlp = Vecchia.nlp(obj, init, box_lower, box_upper)
    (model, solver) = uno(nlp, false; preset="filtersqp", 
                          print_solution=false)
    UnoSolver.uno_get_primal_solution(solver, zeros(length(init)))
  end

end

