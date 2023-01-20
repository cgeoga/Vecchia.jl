module Vecchia

  using Printf, LinearAlgebra, NearestNeighbors, StaticArrays, SparseArrays#, SnoopPrecompile
  using GPMaxlik, JuMP, Ipopt 
  using ForwardDiff
  using JuMP.MathOptInterface
  using ForwardDiff.DiffResults

  const MOI = MathOptInterface

  export nll, vecchia_estimate, em_estimate

  include("warnings.jl")

  include("structstypes.jl")

  include("methods.jl")

  include("utils.jl")

  include("nll.jl")

  include("rcholesky.jl")

  include("sqp.jl")

  include("em.jl")

  include("em_iterator.jl")

  include("ichol.jl")

  #@precompile_all_calls begin
  #  include(@__DIR__()*"/precompile/precompile.jl")
  #end

end 
