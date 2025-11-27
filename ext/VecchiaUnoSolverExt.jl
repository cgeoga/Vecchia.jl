
module VecchiaUnoSolverExt

  using Vecchia, ForwardDiff, NLPModels, UnoSolver

  function Vecchia.optimize(obj::C, init, 
                            solver::Vecchia.NLPModelsSolver{typeof(uno), D};
                            box_lower=fill(1e-5, length(init)),
                            box_upper=fill(100.0,length(init))) where{C,D}
    nlp = Vecchia.nlp(obj, init; box_lower, box_upper)
    (model, solver) = uno(nlp, false; print_solution=false, solver.opts...)
    UnoSolver.uno_get_primal_solution(solver, zeros(length(init)))
  end

end

