
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

struct CondLogLikBuf{D,T}
  buf_pp::Matrix{T}
  buf_cp::Matrix{T}
  buf_cc::Matrix{T}
  buf_cdat::Matrix{T}
  buf_mdat::Matrix{T}
  buf_cpts::Vector{SVector{D,Float64}}
end

function cnllbuf(::Val{D}, ::Val{Z}, ndata, cpts_sz, pts_sz) where{D,Z}
  buf_pp = Array{Z}(undef,  pts_sz,  pts_sz)
  buf_cp = Array{Z}(undef, cpts_sz,  pts_sz)
  buf_cc = Array{Z}(undef, cpts_sz, cpts_sz)
  buf_cdat = Array{Z}(undef, cpts_sz, ndata)
  buf_mdat = Array{Z}(undef,  pts_sz, ndata)
  buf_cpts = Array{SVector{D,Float64}}(undef, cpts_sz)
  CondLogLikBuf{D,Z}(buf_pp, buf_cp, buf_cc, buf_cdat, buf_mdat, buf_cpts)
end

struct CondRCholBuf{D,T}
  buf_pp::Matrix{T}
  buf_cp::Matrix{T}
  buf_cc::Matrix{T}
  buf_cpts::Vector{SVector{D,Float64}}
end

function crcholbuf(::Val{D}, ::Val{Z}, cpts_sz, pts_sz) where{D,Z}
  buf_pp = Array{Z}(undef,  pts_sz,  pts_sz)
  buf_cp = Array{Z}(undef, cpts_sz,  pts_sz)
  buf_cc = Array{Z}(undef, cpts_sz, cpts_sz)
  buf_cpts = Array{SVector{D,Float64}}(undef, cpts_sz)
  CondRCholBuf{D,Z}(buf_pp, buf_cp, buf_cc, buf_cpts)
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

