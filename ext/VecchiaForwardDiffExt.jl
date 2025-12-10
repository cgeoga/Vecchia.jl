
module VecchiaForwardDiffExt

  using Printf, LinearAlgebra, Vecchia, ForwardDiff

  # this is just temporary for the Uno issue.
  function temporary_print(x, case)
    if case == :primal
      @printf "Requesting primal   at [%1.15e, %1.15e, %1.15e]..." x[1] x[2] x[3]
    elseif case == :gradient
      @printf "Requesting gradient at [%1.15e, %1.15e, %1.15e]..." x[1] x[2] x[3]
    elseif case == :hessian
      @printf "Requesting hessian  at [%1.15e, %1.15e, %1.15e]..." x[1] x[2] x[3]
    end
  end

  struct EvaluationResult
    primal::Float64
    gradient::Union{Nothing, Vector{Float64}}
    hessian::Union{Nothing, Symmetric{Float64, Matrix{Float64}}}
  end 

  struct CachingADWrapper{F}
    fn::F
    cache::Dict{Vector{Float64}, EvaluationResult}
  end

  function Vecchia.adcachewrapper(fn::F) where{F}
    cache = Dict{Vector{Float64}, EvaluationResult}()
    CachingADWrapper(fn, cache)
  end

  function Vecchia._primal(cw::CachingADWrapper{F}, x) where{F}
    temporary_print(x, :primal)
    haskey(cw.cache, x) && return (println("found.") ; cw.cache[x].primal)
    println("not found.")
    primal = cw.fn(x)
    cw.cache[x] = EvaluationResult(primal, nothing, nothing)
    primal
  end

  function Vecchia._gradient(cw::CachingADWrapper{F}, x) where{F}
    temporary_print(x, :gradient)
    if haskey(cw.cache, x)
      x_res = cw.cache[x]
      isnothing(x_res.gradient) || (println("found."); return x_res.gradient)
    end
    println("not found.")
    res = DiffResults.GradientResult(x)
    ForwardDiff.gradient!(res, cw.fn, x)
    store = EvaluationResult(DiffResults.value(res), 
                             DiffResults.gradient(res), 
                             nothing)
    cw.cache[x] = store
    DiffResults.gradient(res)
  end

  function Vecchia._hessian(cw::CachingADWrapper{F}, x) where{F}
    temporary_print(x, :hessian)
    if haskey(cw.cache, x)
      x_res = cw.cache[x]
      isnothing(x_res.hessian) || (println("found"); return x_res.hessian)
    end
    println("not found.")
    res = DiffResults.HessianResult(x)
    ForwardDiff.hessian!(res, cw.fn, x)
    store = EvaluationResult(DiffResults.value(res), 
                             DiffResults.gradient(res), 
                             Symmetric(DiffResults.hessian(res)))
    cw.cache[x] = store
    DiffResults.hessian(res)
  end

end

