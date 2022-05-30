module Vecchia

  using LinearAlgebra, NearestNeighbors, StaticArrays, SparseArrays
  using FLoops, BangBang, MicroCollections
  using LoopVectorization 

  export nll

  include("structstypes.jl")

  include("methods.jl")

  include("utils.jl")

  include("nll.jl")

  include("interfaces.jl")

  include("precision.jl")

  include("rcholesky.jl")

end 
