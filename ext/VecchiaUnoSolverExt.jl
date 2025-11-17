
module VecchiaUnoSolverExt

  using Vecchia, ForwardDiff, UnoSolver, NLPModels

  #function Vecchia.optimize(obj, init, solver::Vecchia.UnoNLPSolver;
  function Vecchia.optimize(obj, init, solver;
                            box_lower=fill(0.0, length(init)),
                            box_upper=fill(Inf, length(init)))
    nlp = Vecchia.nlp(obj, init, box_lower=box_lower, box_upper=box_upper)
    (model, solver) = uno(nlp, preset=solver.preset, print_solution=false,
                          operator=false)
    UnoSolver.uno_get_primal_solution(solver, zeros(length(init)))
  end

end

