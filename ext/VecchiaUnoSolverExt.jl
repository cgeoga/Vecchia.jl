
module VecchiaUnoSolverExt

  using Vecchia, ForwardDiff, UnoSolver, NLPModels

  function Vecchia.optimize(obj, init, solver::Vecchia.UnoNLPSolver;
                            box_lower=fill(0.0, length(init)),
                            box_upper=fill(Inf, length(init)))
    nlp = Vecchia.nlp(obj, init, box_lower=box_lower, box_upper=box_upper)
    return nlp
    (model, solver) = uno(nlp, false; preset="filtersqp", 
                          print_solution=false, operator_available=false)
    UnoSolver.uno_get_primal_solution(solver, zeros(length(init)))
  end

end

