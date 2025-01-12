module Vecchia

  # Serious/complex dependencies:
  using JuMP

  # Lightweight dependencies:
  using NearestNeighbors, StaticArrays, Ipopt, GPMaxlik, ForwardDiff#, PrecompileTools

  # Effectively or literally standard library dependencies:
  using Printf, LinearAlgebra, SparseArrays

  # Not new dependencies, just bringing in modules in the namespace of existing
  # dependencies:
  using MathOptInterface
  using ForwardDiff.DiffResults
  const MOI = MathOptInterface

  export nll, vecchia_estimate, em_estimate

  include("warnings.jl")

  include("structstypes.jl")

  include("utils.jl")

  include("nll.jl")

  include("rcholesky.jl")

  include("sqp.jl")

  include("errormatrix.jl")

  include("em.jl")

  include("em_iterator.jl")

  include("ichol.jl")

  include("predict_sim.jl")

  #@compile_workload begin
  #  include(@__DIR__()*"/precompile/precompile.jl")
  #end

end 
