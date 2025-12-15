
module Vecchia

  using Random, LinearAlgebra, SparseArrays, StaticArrays, HNSW, Distances

  # from Distances.jl
  export Euclidean, Haversine

  include("vecchia_config.jl")
  export VecchiaApproximation, Sorted1D, RandomOrdering, NoPermutation, SinglePredictionSets, KNNConditioning

  include("utils.jl")

  include("config_constructors.jl")
  export knnconfig

  include("tiles.jl")

  include("nll.jl")

  include("rcholesky.jl")
  export rchol, rchol_preconditioner

  include("extensions.jl")
  export NLPModelsSolver, vecchia_estimate, vecchia_estimate_nugget

  include("predict.jl")
  export predict

end 

