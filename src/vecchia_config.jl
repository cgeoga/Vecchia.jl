
abstract type VecchiaApproximation{M,D,F} end

struct ZeroMean end
(zm::ZeroMean)(t::T) where{T} = zero(eltype(T))
function (zm::ZeroMean)(t::T, params::P) where{T,P} 
  zero(promote_type(eltype(T), eltype(P)))
end
nparams(zm::ZeroMean) = 0

struct ChunkedVecchiaApproximation{M,D,F} <: VecchiaApproximation{M,D,F}
  meanfun::M
  kernel::F
  data::Vector{Matrix{Float64}} # will be a dummy placeholder if not given
  pts::Vector{Vector{SVector{D, Float64}}}
  condix::Vector{Vector{Int64}} 
  perm::Vector{Int64}
end

# TODO (cg 2025/12/14 22:43): write optimized methods for this object. In
# principle, it should have one less layer of indirection in the pts and data
# access. I think a much more streamlined but equally optimized version of nll
# could also be implemented.
#
# Actually, for the nll, would just need to make CondLogLikBuf into
# ChunkedCondLogLikBuf and then make a separate SingletonCondLogLikBuf. Then
# write a new allocator, a couple other glue methods, and then an optimized
# cnll_str method. But that should all be quite easy.
#
# For rchol, probably even easier to optimize. 
#
# Maybe worth trying the Channel-based threading model for both because it would
# be so simple?
struct SingletonVecchiaApproximation{M,P,F} <: VecchiaApproximation{M,P,F}
  meanfun::M
  kernel::F
  data::Union{Nothing, Matrix{Float64}} 
  pts::Vector{P}
  condix::Vector{Vector{Int64}} 
  perm::Vector{Int64}
end

function Base.display(V::ChunkedVecchiaApproximation)
  println("Chunked Vecchia configuration with:")
  println("  - prediction set size:   $(chunksize(V))")
  println("  - conditioning set size: $(blockrank(V))")
end

function Base.display(V::SingletonVecchiaApproximation)
  println("Singleton Vecchia configuration with:")
  println("  - conditioning set size: $(blockrank(V))")
end

#
# Abstract type umbrellas and offered options here. The intent of the redesign
# is to make it easy to define a new option for any of these categories as
# necessary with a small addition here, and then in an extension implement the
# constructors as appropriate.
#

abstract type PointEnumeration end
struct Sorted1D <: PointEnumeration end
struct RandomOrdering{R} <: PointEnumeration 
  rng::R
end
struct NoPermutation <: PointEnumeration  end
struct HilbertCurveOrdering <: PointEnumeration end

RandomOrdering() = RandomOrdering(Random.default_rng())

# TODO (cg 2025/12/14 16:26): add the ChunkedPredictionSets with
# NearestNeighbors.jl in an extension.
abstract type PredictionSetDesign end
struct SingletonPredictionSets <: PredictionSetDesign end

struct ChunkedPredictionSets <: PredictionSetDesign
  chunksize::Int
end

abstract type ConditioningSetDesign end
struct KNNConditioning{M} <: ConditioningSetDesign 
  k::Int64
  metric::M
end

KNNConditioning(k::Int64) = KNNConditioning(k, Euclidean())

struct KPastIndicesConditioning <: ConditioningSetDesign
  k::Int
end


"""
  `VecchiaApproximation(pts, kernel, data=nothing; meanfun=ZeroMean(), ordering::PointEnumeration, predictionsets::PredictionSetDesign, conditioning::ConditioningSetDesign)

The primary object of this package that specifies and prepares the point ordering, prediction set, and conditioning set design. Arguments are:

- `pts::Vector{P}`: the locations at which your process was measured.
- `kernel::K`: your covariance function, which implements the method `(kernel)(x::P, y::P, params)`.
- `data::Union{Nothing, Matrix{Float64}}`: an option to provide data. If you are fitting a model, provide your data as a matrix where each **column** is an iid replicate. If you are just building a sparse precision matrix, you do not need to adjust this kwarg.

Keyword arguments, which specify details of the approximation, are:

- `meanfun`: a function (or functor) with signature `meanfun(x::P, params)` that returns the mean `E[ your_gp(x) ]`. Note that you have two options evaluating the log-likelihood, one that uses the `Parameters` object that splits mean and covariance function parameters for you (so that each of the two functions can index their own parameter list the natural way) and one that just passes in a flat `Vector{T}`, in which case you manage the indexing yourself. See the example files, the README, and below for more information.
- `ordering::PointEnumeration`: an option indicating how, if at all, you would like points to be reordered. The default option is `RandomOrdering()` in 2+D and canonical sorting in 1D, but there is also `NoPermutation()`. Extensions may provide additional routines.
- `predictionsets::PredictionSetDesign`: an option indicating whether you want to predict single values (`SingletonPredictionSets()`, the default) or chunked prediction sets (not currently available, as legacy code has been removed but not yet ported to an extension).
- `conditioning::ConditioningSetDesign`: an option indicating how you want to determine conditioning sets. The default is `KNNConditioningSets(10)`. Additional methods may be made available via package extensions.

After it is created, it also is a functor that can be used to evaluate the approximate. You have two options for doing so. If `appx::VecchiaApproximation` is your specified model, then you can evaluate it with

```julia
appx(params::Vector{Float64}) # both mean and covariance functions get the whole parameter vector
appx(params::Vector{Float64}; cov_param_ixs, mean_param_ixs) # mean gets `params[mean_param_ixs]`, similar for cov. fun.
```

These options are specifcally so that you can choose to manually handle parameter indexing of one big vector in your mean and covariance function (as in, one of those functions presumably isn't using indices `1:something`) or choose to split them up so that each function can use one-based indexing with the parameters.
"""
function VecchiaApproximation end

"""
`Parameters(;cov_params, mean_params)`

This is an optional structure for you to specify mean and covariance function parameters separately and hand that information to the `vecchia_estimate` routine via the `init` argument. It is entirely optional, and you can instead also pass in `init::Vector{Float64}`---but then you will be responsible for making sure that the mean and covariance functions index the raw flat parameter vector appropriately.
"""
Base.@kwdef struct Parameters
  cov_params::Vector{Float64}
  mean_params::Vector{Float64}
end

Base.length(p::Parameters)  = length(p.cov_params) + length(p.mean_params)
Base.collect(p::Parameters) = vcat(p.cov_params, p.mean_params)

