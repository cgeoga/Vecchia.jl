
# This file assumes that example_setup.jl has been run already!!!

# nll and derivatives in the format Ipopt wants. This is also a simple template
# for exploiting redundant computations in ForwardDiff.
#
# This file expects example_setup.jl to have been run first, as well as the
# "vecc" object to have been created already.

using ForwardDiff
const RESULTS = Dict{UInt64, DiffResults.MutableDiffResult}()
  
__nll(p) = nll(vecc, p)
function addkey!(p)
  newres = DiffResults.HessianResult(p)
  ForwardDiff.hessian!(newres, __nll, p)
  RESULTS[hash(p)] = newres
end

function _nll(p) 
  haskey(RESULTS, hash(p)) && DiffResults.value(RESULTS[hash(p)])
  __nll(p)
end

function grad!(p, store)
  if haskey(RESULTS, hash(p))
    DiffResults.gradient(RESULTS[hash(p)])
    store .= DiffResults.gradient(RESULTS[hash(p)])
  else
    addkey!(p)
    store .= DiffResults.gradient(RESULTS[hash(p)])
  end
end

function hess(p) 
  haskey(RESULTS, hash(p)) && DiffResults.hessian(RESULTS[hash(p)])
  addkey!(p)
  DiffResults.hessian(RESULTS[hash(p)])
end

