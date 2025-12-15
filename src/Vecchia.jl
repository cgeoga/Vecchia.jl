
module Vecchia

  using Printf, Random, LinearAlgebra, SparseArrays, StaticArrays, HNSW, Distances

  # from Distances.jl
  export Euclidean, Haversine

  include("warnings.jl")

  include("vecchia_config.jl")
  export VecchiaApproximation, Sorted1D, RandomOrdering, NoPermutation, SinglePredictionSets, KNNConditioning

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

  include("predict.jl")
  export predict

  #include("predict_sim.jl")

end 

