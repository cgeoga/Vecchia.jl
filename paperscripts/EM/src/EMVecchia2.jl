
module EMVecchia2

  using LinearAlgebra, Vecchia, SparseArrays, Ipopt, StaticArrays
  using GPMaxlik, FiniteDiff, KNITRO, ForwardDiff
  import Vecchia: ipopt_hessian, VecchiaConfig

  const DEF_IPOPT_KWARGS = Dict(:max_iter=>100, :print_level=>5, :tol=>1e-5,
                                :box_lower=>1e-3, :box_upper=>1e22, :sb=>"yes",
                                :try_restart=>0, :accept_every_trial_step=>"no", 
                                :alpha_red_factor=>0.5)

  Base.floatmax(x::ForwardDiff.Dual{T,V,N}) where{T,V,N} = floatmax(V)

  include("utils.jl")

  include("hutchpp.jl")

  include("em.jl")

  include("em_debugging.jl")

  include("iterator.jl")

  include("knitro_optimize.jl")

  include("knitro_zerofind.jl")

end
