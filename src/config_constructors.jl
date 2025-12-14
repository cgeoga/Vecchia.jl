
#=
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
  #dat_out = map(vec_of_vecs_to_matrows, Iterators.partition(data_p, chunksize))
  dat_out = hcat.(Iterators.partition(data_p, chunksize))
  # re-order the centroids of THOSE leaves.
  centroids = map(_mean, pts_out)
  c_ix_dict = Dict(zip(centroids, eachindex(centroids)))
  c_tree    = KDTree(centroids)
  perm      = [c_ix_dict[x] for x in c_tree.data]
  @assert sort(perm) == collect(eachindex(centroids)) "debug test."
  pts_out   = pts_out[perm]
  dat_out   = dat_out[perm]
  # Create the conditioning meta-indices for the chunks.
  condix  = [collect(max(1, (j-blockrank)):(j-1)) for j in eachindex(pts_out)]
  (H,D,F) = (eltype(data), length(first(pts)), typeof(kfun))
  VecchiaApproximation{H,D,F}(kfun, dat_out, pts_out, condix)
end
=#

"""
`knnconfig(data::Matrix{Float64}, pts::Vector{SVector{D,Float64}}, k::Union{Int64, Vector{Int64}}, kernel::Function; randomize::Bool=false, metric=Distances.Euclidean())`

A method for producing a `VecchiaApproximation`, which fully specifies a Vecchia
approximation and implements highly optimized methods for evaluating the 
negative log-likelihood. Arguments are:

- `data`: a matrix where each **column** represents an iid copy of the GP you are modeling.
- `pts`: a vector where each entry `pts[j]` gives the location at `data[j,:]` was measured.
- `k`: a single `Int` or `Vector{Int}` specifying the number of conditioning points to use for each prediction problem.
- `kernel`: your covariance function. **NOTE:**  this function can be arbitrary, but it *must* have the method `kernel(x::SVector{D,Float64}, y::SVector{D,Float64}, params)`.

Optional kwargs are:

- `randomize`: whether to randomly shuffle the order of your points. For lattice data, this is recommended for improved accuracy.
- ` metric`: the metric used for determining nearest neighbors. You probably won't have to touch this unless you are on a sphere, in which case you should pass in `Vecchia.Haversine()` instead.
"""
function knnconfig end

function knnconfig(data, pts, kv, kfun; 
                   randomize=false, metric=Euclidean())
  if randomize
    p = Random.randperm(length(pts))
    return knnconfig(data[p,:], pts[p], kv[p], kfun; randomize=false)
  end
  condix = [Int64[]]
  tree   = HierarchicalNSW(pts; metric=metric)
  for j in 2:length(pts)
    add_to_graph!(tree, [j-1])
    ptj  = pts[j]
    idxs = Int64.(knn_search(tree, pts[j], min(j-1, kv[j]))[1])
    push!(condix, sort(idxs))
  end
  pts = [[x] for x in pts]
  dat = collect.(permutedims.(eachrow(data)))
  VecchiaApproximation(kfun, dat, pts, condix)
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
  VecchiaApproximation(kfun, hcat.(eachrow(_data)), [[SA[x]] for x in _pts], cix)
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

#=
function maximinconfig(::Val{D}, data, ptsm::Matrix{Float64}, 
                       rho::Float64, kernel) where{D}
  (P, ℓ, sn) = ordering_and_sparsity_pattern(ptsm, rho)
  condix = supernodes_to_condix(size(ptsm, 2), sn)
  _data  = data[P,:]
  _pts   = SVector{D,Float64}.(eachcol(ptsm[:,P]))
  VecchiaApproximation(kernel, hcat.(eachrow(_data)), [[x] for x in _pts], condix)
end

function maximinconfig(data, pts::Vector{SVector{D,Float64}}, 
                       rho::Float64, kernel) where{D}
  ptsm = reduce(hcat, pts)
  maximinconfig(Val(D), data, ptsm, rho, kernel)
end
=#
