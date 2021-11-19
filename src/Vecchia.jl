module Vecchia

  using LinearAlgebra, NearestNeighbors, StaticArrays, SparseArrays
  using LoopVectorization 
  using FLoops, BangBang, MicroCollections

  export nll

  include("structstypes.jl")

  include("utils.jl")

  include("nll.jl")

  include("interfaces.jl")

  include("precision.jl")

end 
