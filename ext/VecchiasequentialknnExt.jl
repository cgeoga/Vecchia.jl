
module VecchiasequentialknnExt

  using Vecchia, sequentialknn_jll
  using Vecchia.StaticArrays
  using Vecchia.Distances

  function Vecchia.sknn_conditioningsets(pts::Vector{SVector{D,Float64}}, 
                                         design::KNNConditioning{Euclidean}) where{D}
    k   = design.k
    knn = zeros(UInt64, (k, length(pts)))
    @ccall libsequentialknn.sequential_knn(knn::Ptr{UInt64}, pts::Ptr{Float64},
                                           length(pts)::Csize_t, D::Csize_t,
                                           k::Csize_t)::Cvoid
    map(enumerate(eachcol(knn))) do (j, cj)
      c = Int64.(cj)[1:min(j-1, k)]
      sort!(c)
      c
    end
  end

end

