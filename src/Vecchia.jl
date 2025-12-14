
module Vecchia

  using Printf, Random, LinearAlgebra, SparseArrays, NearestNeighbors, StaticArrays, HNSW, Distances

  export vecchia_estimate, vecchia_estimate_nugget, knnconfig, rchol, NLPModelsSolver

  include("warnings.jl")

  include("vecchia_config.jl")
  export VecchiaConfig

  include("utils.jl")

  include("config_constructors.jl")

  include("tiles.jl")

  include("nll.jl")

  include("rcholesky.jl")

  include("em.jl")

  #include("predict_sim.jl")

end 

