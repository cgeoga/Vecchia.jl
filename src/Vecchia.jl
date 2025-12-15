
module Vecchia

  using Printf, Random, LinearAlgebra, SparseArrays, NearestNeighbors, StaticArrays, HNSW, Distances

  include("warnings.jl")

  include("vecchia_config.jl")
  export VecchiaApproximation

  include("utils.jl")

  include("config_constructors.jl")
  export knnconfig

  include("tiles.jl")

  include("nll.jl")

  include("rcholesky.jl")
  export rchol

  include("em.jl")

  include("extensions.jl")
  export NLPModelsSolver, vecchia_estimate, vecchia_estimate_nugget

  #include("predict_sim.jl")

end 

