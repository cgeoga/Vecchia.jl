
struct CovarianceTiles{T}
  store::Dict{Tuple{Int64, Int64}, Matrix{T}}
end

function Base.getindex(tiles::CovarianceTiles{T}, j::Int64, k::Int64) where{T}
  store = tiles.store
  haskey(store, (j,k)) && return store[(j,k)]
  haskey(store, (k,j)) && return store[(k,j)]'
  throw(error("No tile available for pair ($j, $k)."))
end

function alloc_tiles(pts, condix, ::Val{H}) where{H}
  req_pairs = Set{Tuple{Int64, Int64}}()
  sizehint!(req_pairs, 2*length(condix)*maximum(length, condix))
  for j in eachindex(condix)
      push!(req_pairs, (j, j))
    cj = condix[j]
    for (ixk, k) in enumerate(cj)
      push!(req_pairs, (j, k))
      for l in 1:ixk
        push!(req_pairs, (k, cj[l]))
      end
    end
  end
  req_pairs = collect(req_pairs)
  bufs = map(req_pairs) do jk
    (ptj, ptk) = (pts[jk[1]], pts[jk[2]])
    (lj, lk)   = (length(ptj), length(ptk))
    buf = Array{H}(undef, lj, lk)
  end
  (req_pairs, bufs) 
end

function update_tile_buffers!(tiles::CovarianceTiles{T}, pts, kernel::F, 
                              indices::Vector{Tuple{Int64, Int64}}, p) where{T,F}
  store = tiles.store
  m = cld(length(store), Threads.nthreads())
  index_chunks = Iterators.partition(eachindex(indices), m)
  @sync for ixs in index_chunks
    Threads.@spawn for j in ixs
      (_j, _k)   = indices[j]
      (ptj, ptk) = (pts[_j], pts[_k])
      buf = store[(_j,_k)]
      updatebuf!(buf, ptj, ptk, kernel, p, skipltri=false)
    end
  end
  tiles
end

function build_tiles(pts, condix, kernel::F, p, ::Val{H}) where{F,H}
  (req_pairs, bufs) = alloc_tiles(pts, condix, Val(H))
  tiles = CovarianceTiles(Dict(zip(req_pairs, bufs)))
  update_tile_buffers!(tiles, pts, kernel, req_pairs, p)
end

function build_tiles(cfg::VecchiaApproximation{H,D,F}, p, ::Val{Z}) where{H,D,F,Z}
  build_tiles(cfg.pts, cfg.condix, cfg.kernel, p, Val(Z))
end

