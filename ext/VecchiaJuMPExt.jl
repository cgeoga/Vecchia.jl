
module VecchiaJuMPExt

  using Vecchia, JuMP

  struct WrapSplatted{F} <: Function
    f::F
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

  function Vecchia.optimize(obj, init, solver;
                            box_lower=fill(0.0, length(init)),
                            box_upper=fill(Inf, length(init)))
    p     = length(init)
    objs  = WrapSplatted(obj)
    model = Model(solver)
    @variable(model, box_lower[i] <= params[i=1:p] <= box_upper[i], start=init[i])
    @operator(model, op_obj, p, objs)
    @objective(model, Min, op_obj(params...))
    optimize!(model)
    JuMP.is_solved_and_feasible(model) || @warn "Optimization returned un-successful exit code."
    value.(params)
  end

end

