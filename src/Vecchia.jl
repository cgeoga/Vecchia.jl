module Vecchia

  using LinearAlgebra, NearestNeighbors, StaticArrays, SparseArrays
  using GPMaxlik, StandaloneIpopt
  using ForwardDiff
  using ForwardDiff.DiffResults

  export nll, vecchia_estimate, em_estimate

  include("structstypes.jl")

  include("methods.jl")

  include("utils.jl")

  include("nll.jl")

  include("rcholesky.jl")

  include("em.jl")

  include("em_iterator.jl")

end 
