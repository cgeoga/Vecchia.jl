
const PrecisionPiece{T} = Tuple{Vector{Int64}, Vector{Int64}, Vector{T}}

abstract type AbstractVecchiaConfig{H,D,F} end

# TODO (cg 2021/04/25 13:06): should these fields chunksize and blockrank be in
# here? Arguably the are redundant and encoded in the data/pts/condix values.
# And having them sort of provides a dangerously easy option to not check and
# make sure what those sizes really need to be.
struct VecchiaConfig{H,D,F} <: AbstractVecchiaConfig{H,D,F}
  chunksize::Int64
  blockrank::Int64
  kernel::F
  data::Vector{Matrix{H}}
  pts::Vector{Vector{SVector{D, Float64}}}
  condix::Vector{Vector{Int64}} 
end

function Base.display(V::VecchiaConfig)
  println("Vecchia configuration with:")
  println("  - chunksize:  $(V.chunksize)")
  println("  - block rank: $(V.blockrank)")
  println("  - data size:  $(sum(x->size(x,1), V.data))")
  println("  - nsamples:   $(size(V.data[1], 2))")
end

# TODO (cg 2021/04/25 13:06): should these fields chunksize and blockrank be in
# here? Arguably the are redundant and encoded in the data/pts/condix values.
# And having them sort of provides a dangerously easy option to not check and
# make sure what those sizes really need to be.
struct ScalarVecchiaConfig{H,D,F} <: AbstractVecchiaConfig{H,D,F}
  chunksize::Int64
  blockrank::Int64
  kernel::F
  data::Vector{Matrix{H}}
  pts::Vector{Vector{Float64}}
  condix::Vector{Vector{Int64}} 
end

function Base.display(V::ScalarVecchiaConfig)
  println("Scalarized Vecchia configuration with:")
  println("chunksize:  $(V.chunksize)")
  println("block rank: $(V.blockrank)")
  println("data size:  $(sum(x->size(x,1), V.data))")
  println("nsamples:   $(size(V.data[1], 2))")
end

struct RCholesky{T}
  diagonals::Vector{UpperTriangular{T,Matrix{T}}}
  odiagonals::Vector{Matrix{T}}
  condix::Vector{Vector{Int64}}
  idxs::Vector{UnitRange{Int64}} 
  is_instantiated::Vector{Bool}
end

# TODO (cg 2022/05/30 12:10): make this more information.
Base.display(U::RCholesky{T}) where{T} = println("RCholesky{$T}")
Base.display(Ut::Adjoint{T,RCholesky{T}}) where{T} = println("Adjoint{$T, RCholesky{$T}}")

struct MemoizedKernel{F,T}
  cache::Dict{Tuple{UInt64, UInt64},T}
  phash::UInt64
  kernel::F
end

# Not good code or anything. Just a quick and dirty way to make a Vecchia object
# with a KD-tree to choose the conditioning points.
function kdtreeconfig(data, pts, chunksize, blockrank, kfun)
  (data isa Vector) && return kdtreeconfig(hcat(data), pts, chunksize, blockrank, kfun)
  # Make a KDTree of the points with a certain leaf size
  tree    = KDTree(pts, leafsize=chunksize)
  # re-order the data accordingly:
  _d      = Dict(zip(pts, eachrow(data)))
  data_p  = [_d[xj] for xj in tree.data]
  # Get the data in chunks of size $chunksize.
  pts_out = map(copy, Iterators.partition(tree.data, chunksize))
  # TODO (cg 2022/04/21 16:17): The inner function here (vec_of_...), really
  # could use some optimization and is kind of just a stand-in at the moment. So
  # long as this constructor isn't in the hot loop of an optimizer it's probably
  # fine as-is, but just a note that there is some perf on the table here.
  dat_out = map(vec_of_vecs_to_matrows, Iterators.partition(data_p, chunksize))
  # re-order the centroids of THOSE leaves.
  centroids = map(x->sum(x)/length(x), pts_out)
  c_ix_dict = Dict(zip(centroids, eachindex(centroids)))
  c_tree    = KDTree(centroids)
  perm      = [c_ix_dict[x] for x in c_tree.data]
  @assert sort(perm) == collect(eachindex(centroids)) "debug test."
  pts_out   = pts_out[perm]
  dat_out   = dat_out[perm]
  # Create the conditioning meta-indices for the chunks.
  condix  = map(j->cond_ixs(j,blockrank), eachindex(pts_out))
  (H,D,F) = (eltype(data), length(first(pts)), typeof(kfun))
  VecchiaConfig{H,D,F}(min(chunksize, length(first(pts_out))),
                       min(blockrank, length(pts_out)),
                       kfun, dat_out, pts_out, condix)
end

# Even less good code, but here's a simple connection between Vecchia and the
# H-matrix with global Nystrom that I've worked on earlier: add a single
# collection of "global" points to the conditioning set of all subsequent
# points. Pretty easy to do with a couple extra tree constructions.
function nystrom_kdtreeconfig(data, pts, chunksize, blockrank, kfun, nys_size)
  (data isa Vector) && return nystrom_kdtreeconfig(hcat(data), pts, chunksize, 
                                                   blockrank, kfun, nys_size)
  # Make a KDTree of the points with a certain leaf size
  tree   = KDTree(pts, leafsize=chunksize)
  _d     = Dict(zip(pts, eachrow(data)))
  data_p = reshape(reduce(vcat, [_d[xj] for xj in tree.data]), :, size(data,2))
  # Pick the "nystrom points" in the simplest way:
  nys_inds = 1:div(length(pts), nys_size):length(pts)
  nys_pts  = tree.data[nys_inds]
  # now for the rest of the points, do the ordering above:
  _cfg     = kdtreeconfig(data_p, tree.data[setdiff(1:length(pts), nys_inds)],
            chunksize, blockrank, kfun)
  # extract and modify the fields we actually want:
  pts_out  = vcat([nys_pts], _cfg.pts)
  nys_dat  = reshape(reduce(vcat, data_p[nys_inds]), :, size(data, 2))
  dat_out  = pushfirst!(_cfg.data, nys_dat)
  condix   = map(j->cond_ixs(j,blockrank), eachindex(pts_out))
  for j in 2:length(condix)
    cixj = condix[j]
    if !in(1, cixj)
      pushfirst!(cixj, 1)
    end
  end
  (H,D,F) = (eltype(data), length(first(pts)), typeof(kfun))
  VecchiaConfig{H, D,F}(min(chunksize+nys_size, length(first(pts_out))),
                        min(blockrank+1, length(pts_out)),
                        kfun, dat_out, pts_out, condix)
end


function scalarize(v::VecchiaConfig{H,D,F}, scalarized_kernel::G) where{H,D,F,G}
  scalarized_pts = map(x->reduce(vcat, x), v.pts)
  ScalarVecchiaConfig{H,D,G}(v.chunksize,
                             v.blockrank,
                             scalarized_kernel,
                             v.data,
                             scalarized_pts,
                             v.condix)
end

function MemoizedKernel(f, sample_x, p)
  f11 = f(sample_x, sample_x, p)
  ph  = hash(p)
  cache = Dict{Tuple{UInt64, UInt64},typeof(f11)}()
  MemoizedKernel(cache, ph, f)
end

function (k::MemoizedKernel{F,T})(x, y, p) where{F,T}
  _key = (hash(x), hash(y))
  if haskey(k.cache, _key)
    return k.cache[_key]
  else
    _val = k.kernel(x,y,p)
    k.cache[_key] = _val
    return _val
  end
end



