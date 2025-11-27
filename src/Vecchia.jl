
module Vecchia

  using Printf, Random, LinearAlgebra, SparseArrays, NearestNeighbors, StaticArrays, HNSW, Accessors, Distances

  export nll, vecchia_estimate, vecchia_estimate_nugget, em_estimate, PredictionConfig, knnpredict, knnconfig, rchol, NLPModelsSolver

  include("warnings.jl")

  include("structstypes.jl")

  include("utils.jl")

  include("nll.jl")

  include("rcholesky.jl")

  include("errormatrix.jl")

  include("em.jl")

  include("em_iterator.jl")

  include("predict_sim.jl")

end 

