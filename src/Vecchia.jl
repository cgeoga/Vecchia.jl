module Vecchia

  using LinearAlgebra, NearestNeighbors, StaticArrays, SparseArrays
  using FLoops, BangBang, MicroCollections
  #using LoopVectorization 

  export nll

  include("structstypes.jl")

  include("utils.jl")

  include("nll.jl")

  include("interfaces.jl")

  include("precision.jl")

end 
