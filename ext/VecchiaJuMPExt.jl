
module VecchiaJuMPExt

  using Vecchia, JuMP, ForwardDiff

  struct WrapSplatted{F} <: Function
    f::F
  end

  struct Gradient{F} <: Function
    f::WrapSplatted{F}
  end

  struct Hessian{F} <: Function
    f::WrapSplatted{F}
  end

  function (s::WrapSplatted{F})(p) where{F} 
    try
      return s.f(p)
    catch er
      er isa InterruptException && rethrow(er)
      return NaN
    end
  end
  (s::WrapSplatted{F})(p...) where{F} = s(collect(p))

  function (g::Gradient{F})(buf, p) where{F}
    ForwardDiff.gradient!(buf, g.f, p)
  end
  (g::Gradient{F})(buf, p...) where{F} = g(buf, collect(p))

  function (h::Hessian{F})(buf, p) where{F}
    n = length(p)
    _h = ForwardDiff.hessian(h.f, p)
    for i in 1:n, j in 1:i
      buf[i,j] = _h[i,j]
    end
  end
  (h::Hessian{F})(buf, p...) where{F} = h(buf, collect(p))

  function Vecchia.optimize(obj, init, solver;
                            box_lower=fill(0.0, length(init)),
                            box_upper=fill(Inf, length(init)))
    p     = length(init)
    objs  = WrapSplatted(obj)
    model = Model(solver)
    @variable(model, box_lower[i] <= params[i=1:p] <= box_upper[i], start=init[i])
    @operator(model, op_obj, p, objs, Gradient(objs), Hessian(objs))
    @objective(model, Min, op_obj(params...))
    optimize!(model)
    JuMP.is_solved_and_feasible(model) || @warn "Optimization returned un-successful exit code."
    value.(params)
  end

end

