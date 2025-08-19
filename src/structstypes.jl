
const PrecisionPiece{T} = Tuple{Vector{Int64}, Vector{Int64}, Vector{T}}

abstract type AbstractVecchiaConfig{H,D,F} end

struct ErrorKernel{K,E} <: Function
  kernel::K
  error::E
end

struct TRBSolver
  print_level::Int64
end
TRBSolver(;verbose::Bool=true) = TRBSolver(verbose ? 1 : 0)

function optimize end

function vecchia_estimate(cfg, init, solver; kwargs...)
  optimize(cfg, init, solver; kwargs...)
end

function vecchia_estimate_nugget(cfg, init, solver, errormodel; kwargs...)
  nugkernel = Vecchia.ErrorKernel(cfg.kernel, errormodel)
  nugcfg    = Vecchia.VecchiaConfig(nugkernel, cfg.data, cfg.pts, cfg.condix)
  vecchia_estimate(nugcfg, init, solver; kwargs...)
end

function (k::ErrorKernel{K})(x, y, p) where{K}
  k.kernel(x,y,p)+k.error(x,y,p)
end

struct CovarianceTiles{T}
  store::Dict{Tuple{Int64, Int64}, Matrix{T}}
end

function Base.getindex(tiles::CovarianceTiles{T}, j::Int64, k::Int64) where{T}
  store = tiles.store
  haskey(store, (j,k)) && return store[(j,k)]
  haskey(store, (k,j)) && return store[(k,j)]'
  throw(error("No tile available for pair ($j, $k)."))
end

# TODO (cg 2021/04/25 13:06): should these fields chunksize and blockrank be in
# here? Arguably the are redundant and encoded in the data/pts/condix values.
# And having them sort of provides a dangerously easy option to not check and
# make sure what those sizes really need to be.
struct VecchiaConfig{H,D,F} <: AbstractVecchiaConfig{H,D,F}
  kernel::F
  data::Vector{Matrix{H}}
  pts::Vector{Vector{SVector{D, Float64}}}
  condix::Vector{Vector{Int64}} 
end

function Base.display(V::VecchiaConfig)
  println("Vecchia configuration with:")
  println("  - chunksize:  $(chunksize(V))")
  println("  - block rank: $(blockrank(V))")
  println("  - data size:  $(sum(x->size(x,1), V.data))")
  println("  - nsamples:   $(size(V.data[1], 2))")
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

# A piece of a Vecchia approximation with a single-argument method. Split up
# like this because using ReverseDiff.gradient on the thread-parallel nll
# doesn't work, and so breaking it into pieces like this means that I can more
# easily compile tapes for the chunks that would each be evaluated on a single
# thread, and then parallelize the calls to ReverseDiff.gradient!.
#
# see the method definition in ./nll.jl.
struct VecchiaLikelihoodPiece{H,D,F,T}
  cfg::VecchiaConfig{H,D,F}
  buf::CondLogLikBuf{D,T}
  ixrange::UnitRange{Int64}
end

struct PieceEvaluation{H,D,F,T} <: Function
  piece::VecchiaLikelihoodPiece{H,D,F,T}
end

function (c::PieceEvaluation{H,D,F,T})(p) where{H,D,F,T}
  (logdets, qforms) = c.piece(p)
  ndata = size(first(c.piece.cfg.data), 2)
  (ndata*logdets + qforms)/2
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

# Pass this around instead?
struct RCholApplicationBuffer{T}
  bufv::Matrix{T}
  bufm::Matrix{T}
  bufz::Matrix{T}
  out::Matrix{T}
end

function RCholApplicationBuffer(U::RCholesky{T}, ndata::Int64, ::Val{V}) where{T,V}
  Z = promote_type(T, V)
  m = length(U.condix)
  out  = Array{Z}(undef, maximum(j->size(U.odiagonals[j], 2), 1:m), ndata)
  bufz = Array{Z}(undef, maximum(j->size(U.odiagonals[j], 2), 1:m), ndata)
  bufv = Array{Z}(undef, maximum(length, U.condix)*maximum(length, U.idxs), ndata)
  bufm = Array{Z}(undef, maximum(length, U.idxs), ndata)
  RCholApplicationBuffer{Z}(bufv, bufm, bufz, out)
end

# A struct wrapper of a function that handles argument splatting and also
# "converts" errors to NaNs. Very useful when feeding NLPs to JuMP.
struct WrapSplatted{F} <: Function
  f::F
end
function (s::WrapSplatted{F})(p) where{F} 
  try
    return s.f(p)
  catch er
    er isa InterruptException && rethrow(er)
    return NaN
  end
end
(s::WrapSplatted{F})(p...) where{F} = s(collect(p))

# Not good code or anything. Just a quick and dirty way to make a Vecchia object
# with a KD-tree to choose the conditioning points.
function kdtreeconfig(data, pts, chunksize, blockrank, kfun)
  (data isa Vector) && return kdtreeconfig(hcat(data), pts, chunksize, blockrank, kfun)
  size(data, 1) == length(pts) || @warn "Your input data and points don't have the same length. Consider checking your code for mistakes."
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
  centroids = map(_mean, pts_out)
  c_ix_dict = Dict(zip(centroids, eachindex(centroids)))
  c_tree    = KDTree(centroids)
  perm      = [c_ix_dict[x] for x in c_tree.data]
  @assert sort(perm) == collect(eachindex(centroids)) "debug test."
  pts_out   = pts_out[perm]
  dat_out   = dat_out[perm]
  # Create the conditioning meta-indices for the chunks.
  condix  = [cond_ixs(j,blockrank) for j in eachindex(pts_out)]
  (H,D,F) = (eltype(data), length(first(pts)), typeof(kfun))
  VecchiaConfig{H,D,F}(kfun, dat_out, pts_out, condix)
end

function knnconfig(data, pts, blockranks, kfun; 
                   randomize=false, metric=Euclidean())
  if randomize
    p = Random.randperm(length(pts))
    return knnconfig(data[p,:], pts[p], blockranks[p], kfun; randomize=false)
  end
  condix = [Int64[]]
  tree   = HierarchicalNSW(pts; metric=metric)
  for j in 2:length(pts)
    add_to_graph!(tree, [j-1])
    ptj  = pts[j]
    idxs = Int64.(knn_search(tree, pts[j], min(j-1, blockranks[j]))[1])
    push!(condix, sort(idxs))
  end
  pts = [[x] for x in pts]
  dat = collect.(permutedims.(eachrow(data)))
  VecchiaConfig(kfun, dat, pts, condix)
end

function knnconfig(data, pts, m::Int64, kfun; randomize=false, metric=Euclidean())
  knnconfig(data, pts, fill(m, length(pts)), kfun; randomize=randomize, metric=metric)
end

function knnconfig(data, pts::Vector{SVector{1,Float64}}, mv::AbstractVector{Int64}, kfun; 
                   randomize=false, sort=true)
  randomize && @info "randomize=true flag ignored since in 1D a natural ordering is possible."
  _pts  = getindex.(pts, 1)
  sp    = sortperm(_pts)
  _pts  = _pts[sp]
  _data = data[sp,:]
  cix   = [collect(max(1, j-mv[j]):(j-1)) for j in 1:length(pts)]
  VecchiaConfig(kfun, hcat.(eachrow(_data)), [[SA[x]] for x in _pts], cix)
end

function knnconfig(data, pts::Vector{SVector{1,Float64}}, m::Int, kfun; 
                   randomize=false, sort=true)
  knnconfig(data, pts, fill(m, length(pts)), kfun; randomize=randomize, sort=sort) 
end

function knnconfig(data, pts::Vector{Float64}, m::Int, kfun; 
                   randomize=false, sort=true)
  _pts = [SA[x] for x in pts]
  knnconfig(data, _pts, m, kfun; randomize=randomize, sort=sort)
end

# An internal function that takes an existing configuration and _new_ locations
# and values representing landmark/inducing points and returns a new config that
# keeps all the original conditioning points from the input config, but also
# adds thse fsa points as a chunked conditioning point to all of them.
function _fsa_config(_cfg, fsa_pts, fsa_data)
  cfg = deepcopy(_cfg)
  pushfirst!(cfg.pts,  fsa_pts)
  pushfirst!(cfg.data, hcat(fsa_data))
  for j in eachindex(cfg.condix)
    cj = cfg.condix[j]
    !isempty(cj) && (cj .+= 1)
    pushfirst!(cj, 1)
  end
  pushfirst!(cfg.condix, Int64[])
  cfg
end

function fsa_knnconfig(data, pts, mknn, mfsa, kernel; randomize=true)
  tree    = KDTree(pts)
  fsa_ix  = 1:div(length(pts), mfsa):length(pts)
  fsa_pts = pts[fsa_ix]
  fsa_dat = data[fsa_ix]
  res_pts = copy(pts)
  res_dat = copy(data)
  deleteat!(res_pts, fsa_ix)
  deleteat!(res_dat, fsa_ix)
  res_cfg = knnconfig(res_dat, res_pts, mknn, kernel; randomize=randomize)
  _fsa_config(res_cfg, fsa_pts, fsa_dat)
end

