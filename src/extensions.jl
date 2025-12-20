
#
# This file defines dummy objects or functions that will be given meaningful
# methods and operations through extensions.
#

struct NLPModelsSolver{S,T}
  solver::S
  opts::Dict{Symbol, T}
end

"""
`NLPModelsSolver(s; opts...)`

A thin wrapper for representing a solver object that accepts a `NLPModel` object. None are provided in the base of this package, but by loading the extensions (see the README or example files) one can bring them into scope. For example, to fit a model with Ipopt with a fixed number of iterations and a lower tolerance than default, one would do this:
```
using ForwardDiff, NLPModels, NLPModelsIpopt

solver = NLPModelsSolver(ipopt; :max_iter=>100, :tol=>1e-4)
mle    = vecchia_estimate(cfg, init, solver)
```
"""
NLPModelsSolver(s::S; kwargs...) where{S} = NLPModelsSolver(s, Dict(kwargs))

function nlp end

function adcachewrapper end
function _primal        end
function _gradient      end
function _hessian       end

function optimize end

"""
`vecchia_estimate(cfg::VecchiaApproximation, init, solver; kwargs...)`

Find the MLE under the Vecchia approximation specified by `cfg`. Initialization is provided by `init`, and the optimizer is specified by `solver`. See the README or example files for examples of solvers. **NOTE:** you will need to `using` some additional packages to load the extensions that give this function useful methods.
"""
function vecchia_estimate end

function vecchia_estimate(cfg, init, solver; kwargs...)
  optimize(cfg, init, solver; kwargs...)
end

function hnsw_conditioningsets(args...)
  throw(error("Your particular approxmation design problem, either for dimension or non-Euclidean distance metric or both, is not supported by the default method for this configuration (which is restricted to 2-4 dimensions and Euclidean distance). Please `]add` and `using` the optional dependency `HNSW.jl` to load the fallback extension that will cover your use case."))
end

