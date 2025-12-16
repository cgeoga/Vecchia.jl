
abstract type VecchiaApproximation{D,F} end

struct ChunkedVecchiaApproximation{D,F} <: VecchiaApproximation{D,F}
  kernel::F
  data::Union{Nothing, Vector{Matrix{Float64}}}
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
struct SingletonVecchiaApproximation{D,F} <: VecchiaApproximation{D,F}
  kernel::F
  data::Union{Nothing, Matrix{Float64}}
  pts::Vector{SVector{D, Float64}}
  condix::Vector{Vector{Int64}} 
  perm::Vector{Int64}
end

function Base.display(V::ChunkedVecchiaApproximation)
  println("Chunked Vecchia configuration with:")
  println("  - prediction set size:   $(chunksize(V))")
  println("  - conditioning set size: $(blockrank(V))")
end

function Base.display(V::SingletonVecchiaApproximation)
  println("Chunked Vecchia configuration with:")
  println("  - prediction set size:   $(chunksize(V))")
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

RandomOrdering() = RandomOrdering(Random.default_rng())

# TODO (cg 2025/12/14 16:26): add the ChunkedPredictionSets with
# NearestNeighbors.jl in an extension.
abstract type PredictionSetDesign end
struct SingletonPredictionSets <: PredictionSetDesign end

abstract type ConditioningSetDesign end
struct KNNConditioning{M} <: PredictionSetDesign 
  k::Int64
  metric::M
end

KNNConditioning(k::Int64) = KNNConditioning(k, Euclidean())


"""
  `VecchiaApproximation(pts, kernel, data=nothing; ordering::PointEnumeration, predictionsets::PredictionSetDesign, conditioning::ConditioningSetDesign)

The primary object of this package that specifies and prepares the point ordering, prediction set, and conditioning set design. Arguments are:

- `pts::Vector{SVector{D,Float64}}`: the locations at which your process was measured.
- `kernel::K`: your covariance function, which implements the method `(kernel)(x::SVector{D,Float64}, y::SVector{D,Float64}, params)`.
- `data::Union{Nothing, Matrix{Float64}}`: an option to provide data. If you are fitting a model, provide your data as a matrix where each **column** is an iid replicate. If you are just building a sparse precision matrix, you do not need to adjust this kwarg.

Keyword arguments, which specify details of the approximation, are:

- `ordering::PointEnumeration`: an option indicating how, if at all, you would like points to be reordered. The default option is `RandomOrdering()` in 2+D and canonical sorting in 1D, but there is also `NoPermutation()`. Extensions may provide additional routines.
- `predictionsets::PredictionSetDesign`: an option indicating whether you want to predict single values (`SingletonPredictionSets()`, the default) or chunked prediction sets (not currently available, as legacy code has been removed but not yet ported to an extension).
- `conditioning::ConditioningSetDesign`: an option indicating how you want to determine conditioning sets. The default is `KNNConditioningSets(10)`. Additional methods may be made available via package extensions.
"""
function VecchiaApproximation end

