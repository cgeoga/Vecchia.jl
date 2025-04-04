
module Vecchia

  using Printf, LinearAlgebra, SparseArrays, NearestNeighbors, StaticArrays

  export nll, vecchia_estimate, vecchia_estimate_nugget, em_estimate

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

