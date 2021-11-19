
const PrecisionPiece{T} = Tuple{Vector{Int64}, Vector{Int64}, Vector{T}}

# TODO (cg 2021/04/25 13:06): should these fields chunksize and blockrank be in
# here? Arguably the are redundant and encoded in the data/pts/condix values.
# And having them sort of provides a dangerously easy option to not check and
# make sure what those sizes really need to be.
struct VecchiaConfig{D,F} 
  chunksize::Int64
  blockrank::Int64
  kernel::F
  data::Vector{Vector{Float64}}
  pts::Vector{Vector{SVector{D, Float64}}}
  condix::Vector{Vector{Int64}} 
end

# Not good code or anything. Just a quick and dirty way to make a Vecchia object
# with a KD-tree to choose the conditioning points.
function kdtreeconfig(data, pts, chunksize, blockrank, kfun)
  # Make a KDTree of the points with a certain leaf size
  tree    = KDTree(pts, leafsize=chunksize)
  # re-order the data accordingly:
  _d      = Dict(zip(pts, data))
  data_p  = [_d[xj] for xj in tree.data]
  # Get the data in chunks of size $chunksize.
  pts_out = map(copy, Iterators.partition(tree.data, chunksize))
  dat_out = map(copy, Iterators.partition(data_p, chunksize))
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
  (D,F)   = (length(first(pts)), typeof(kfun))
  VecchiaConfig{D,F}(min(chunksize, length(first(pts_out))),
                     min(blockrank, length(pts_out)),
                     kfun, dat_out, pts_out, condix)
end

# Even less good code, but here's a simple connection between Vecchia and the
# H-matrix with global Nystrom that I've worked on earlier: add a single
# collection of "global" points to the conditioning set of all subsequent
# points. Pretty easy to do with a couple extra tree constructions.
function nystrom_kdtreeconfig(data, pts, chunksize, blockrank, kfun, nys_size)
  # Make a KDTree of the points with a certain leaf size
  tree     = KDTree(pts, leafsize=chunksize)
  _d       = Dict(zip(pts, data))
  data_p   = [_d[xj] for xj in tree.data]
  # Pick the "nystrom points" in the simplest way:
  nys_inds = 1:div(length(pts), nys_size):length(pts)
  nys_pts  = tree.data[nys_inds]
  # now for the rest of the points, do the ordering above:
  _cfg     = kdtreeconfig(data_p, tree.data[setdiff(1:length(pts), nys_inds)],
                          chunksize, blockrank, kfun)
  # extract and modify the fields we actually want:
  pts_out  = vcat([nys_pts], _cfg.pts)
  dat_out  = vcat([data_p[nys_inds]], _cfg.data)
  condix   = map(j->cond_ixs(j,blockrank), eachindex(pts_out))
  for j in 2:length(condix)
    cixj = condix[j]
    if !in(1, cixj)
      pushfirst!(cixj, 1)
    end
  end
  (D,F)   = (length(first(pts)), typeof(kfun))
  VecchiaConfig{D,F}(min(chunksize+nys_size, length(first(pts_out))),
                     min(blockrank+1, length(pts_out)),
                     kfun, dat_out, pts_out, condix)
end

# A new nice little constructor, but not the most thoughtful or robust.
function spacefirstconfig(data, pts, nspace, tchunksize, blockrank, kfun)
  @info "This function assumes points are already sorted in time." maxlog=1
  timechunks_pts = collect.(Iterators.partition(pts,  nspace*tchunksize))
  timechunks_dat = collect.(Iterators.partition(data, nspace*tchunksize))
  condix = map(j->cond_ixs(j,blockrank), eachindex(timechunks_pts))
  (D,F)  = (length(first(pts)), typeof(kfun))
  (timechunks_pts, timechunks_dat)
  VecchiaConfig{D,F}(min(nspace*tchunksize, length(first(timechunks_pts))),
                     min(blockrank, length(timechunks_pts)),
                     kfun, timechunks_dat, timechunks_pts, condix)
end

