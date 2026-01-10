
module Vecchia

  using Random, LinearAlgebra, SparseArrays, StaticArrays, Distances, HNSW

  # from Distances.jl
  export Euclidean, Haversine

  import Base.Threads: @spawn, nthreads

  include("vecchia_config.jl")
  export VecchiaApproximation, ZeroMean, Sorted1D, RandomOrdering, NoPermutation, SinglePredictionSets, KNNConditioning, Parameters

  include("utils.jl")

  include("config_constructors.jl")
  export knnconfig

  include("tiles.jl")

  include("nll.jl")

  include("rcholesky.jl")
  export rchol, rchol_preconditioner, lazy_rchol

  include("extensions.jl")
  export NLPModelsSolver, vecchia_estimate, vecchia_estimate_nugget

  include("predict.jl")
  export predict

end 

