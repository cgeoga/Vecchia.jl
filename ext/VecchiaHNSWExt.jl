
module VecchiaHNSWExt

  using Vecchia, HNSW
  using Vecchia.StaticArrays

  function Vecchia.hnsw_conditioningsets(pts::Vector{SVector{D,Float64}}, 
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

end

