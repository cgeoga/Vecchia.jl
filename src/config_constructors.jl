
function permute_points_and_data(pts::Vector{SVector{D,Float64}}, 
                                 data, ::NoPermutation) where{D}
  if isone(D)
    check_canonical_sorted(pts) || @warn "For points in 1D, there is a very easy canonical sorting. Unless you have a specific reason not to use it, you are better off using that than NoPermutation()." maxlog=1    
  end
  (collect(eachindex(pts)), pts, data) 
end

function permute_points_and_data(pts::Vector{SVector{1,Float64}}, data, ::Sorted1D)
  pts_scalar = getindex.(pts, 1)
  perm       = sortperm(pts_scalar)
  data_perm  = isnothing(data) ? nothing : data[perm,:]
  pts_perm   = pts[perm]
  (perm, pts_perm, data_perm)
end

function permute_points_and_data(pts, data, ordering::RandomOrdering)
  perm      = Random.randperm(ordering.rng, length(pts))
  data_perm = isnothing(data) ? nothing : data[perm,:]
  pts_perm  = pts[perm]
  (perm, pts_perm, data_perm)
end

function hnsw_conditioningsets(pts::Vector{SVector{D,Float64}}, 
                               design::KNNConditioning{M}) where{D,M}
  condix = [Int64[]]
  tree   = HierarchicalNSW(pts; metric=design.metric)
  for j in 2:length(pts)
    add_to_graph!(tree, [j-1])
    ptj  = pts[j]
    idxs = Int64.(knn_search(tree, pts[j], min(j-1, design.k))[1])
    push!(condix, sort(idxs))
  end
  condix
end

function sknn_conditioningsets end

function conditioningsets(pts::Vector{SVector{D,Float64}}, 
                          design::KNNConditioning{M}) where{D,M}
  # if available, using the _extremely_ fast knn conditioning set design
  # routines from the sequentialknn_jll dependency.
  has_ext = length(methods(sknn_conditioningsets)) > 0
  if fastknn_routine_available(pts, design) && has_ext
    @info "Using accelerated sequentialknn_jll methods..." maxlog=1
    sknn_conditioningsets(pts, design)
  # otherwise, HNSW.jl provides a workable fallback that works for _any_
  # distance metric in _any_ dimension. 
  else
    hnsw_conditioningsets(pts, design)
  end
end

function conditioningsets(pts::Vector{SVector{1,Float64}}, 
                          design::KNNConditioning{Euclidean})
  condix = Vector{Vector{Int64}}(undef, length(pts))
  condix[1] = Float64[]
  for j in 2:length(pts)
    condix[j] = collect(max(1, j-design.k):(j-1))
  end
  condix
end


default_ordering(pts::Vector{SVector{1,Float64}}) = Sorted1D()
default_ordering(pts::Vector{SVector{D,Float64}}) where{D} = RandomOrdering()

default_predictionsets() = SingletonPredictionSets()

function default_conditioning(pts::Vector{SVector{1,Float64}}) 
  KNNConditioning(5, Euclidean())
end

function default_conditioning(pts::Vector{SVector{D,Float64}}) where{D}
  KNNConditioning(10, Euclidean())
end

function VecchiaApproximation(pts::Vector{SVector{D,Float64}},
                              kernel::K, data=nothing;
                              meanfun=ZeroMean(),
                              ordering=default_ordering(pts),
                              predictionsets=default_predictionsets(),
                              conditioning=default_conditioning(pts)) where{D,K}
  if predictionsets isa SingletonPredictionSets
    singleton_approximation(pts, kernel, data; meanfun, ordering, conditioning)
  else
    chunked_approximation(pts, kernel, data; meanfun, ordering, 
                          predictionsets, conditioning)
  end
end

function chunk_format_points_and_data(pts, data, ::SingletonPredictionSets)
  pts_str  = [[x] for x in pts]
  data_str = isnothing(data) ? [[NaN;;]] : permutedims.(collect.(eachrow(data))) 
  (pts_str, data_str)
end

function chunked_approximation(pts::Vector{SVector{D,Float64}},
                               kernel::K, data=nothing;
                               meanfun=ZeroMean(),
                               ordering=default_ordering(pts),
                               predictionsets=default_predictionsets(),
                               conditioning=default_conditioning(pts)) where{D,K}
  (perm, _pts_perm, _data_perm) = permute_points_and_data(pts, data, ordering)
  condix = conditioningsets(_pts_perm, conditioning)
  (pts_ch, data_ch) = chunk_format_points_and_data(_pts_perm, _data_perm, 
                                                   predictionsets)
  ChunkedVecchiaApproximation(meanfun, kernel, data_ch, pts_ch, condix, perm)
end

function singleton_approximation(pts::Vector{SVector{D,Float64}},
                                 kernel::K, data=nothing;
                                 meanfun=ZeroMean(),
                                 ordering=ordering,
                                 conditioning=conditioning) where{D,K}
  (perm, pts_perm, data_perm) = permute_points_and_data(pts, data, ordering)
  data_perm = isnothing(data_perm) ? [NaN;;] : hcat(data_perm)
  condix    = conditioningsets(pts_perm, conditioning)
  SingletonVecchiaApproximation(meanfun, kernel, data_perm, pts_perm, condix, perm)
end

