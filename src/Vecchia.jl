module Vecchia

  # TODO (cg 2022/09/08 14:11): I think maybe LV, BangBang, and MicroColl. can
  # be pruned from the dep tree. In general, the precision matrix function
  # should perhaps only be used for debugging. LV definitely does make things
  # faster, but it does really complicate the tree.
  using LinearAlgebra, NearestNeighbors, StaticArrays, SparseArrays
  using FLoops, BangBang, MicroCollections
  using LoopVectorization 
  using GPMaxlik, StandaloneIpopt
  using ForwardDiff
  using ForwardDiff.DiffResults

  export nll, vecchia_estimate

  include("structstypes.jl")

  include("methods.jl")

  include("utils.jl")

  include("nll.jl")

  include("precision.jl")

  include("rcholesky.jl")

  include("em.jl")

  include("em_iterator.jl")

end 
