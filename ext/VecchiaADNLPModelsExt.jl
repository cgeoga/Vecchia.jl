
module VecchiaADNLPModelsExt

  using Vecchia, ADNLPModels

  import Vecchia.UnoNLPSolver

  function Vecchia.nlp(cfg::C, init; 
                       box_lower=fill(0.0, length(init)),
                       box_upper=fill(Inf, length(init)))
    ADNLPModel(obj, init, box_lower, box_upper; backend=:generic)
  end

end

