
"""
`PredictionConfig(cfg::VecchiaConfig, pred_pts, ncondition; [fixed_use_indices=nothing, separable=false])`

A structure representing a Vecchia-accelerated prediction problem. Internally, conditioning
data and its conditioning set (from `cfg`) are joined with the new prediction points, which 
are put at the end so that there is the greatest pool of conditioning points to select from.

ARGUMENTS:

- `cfg::VecchiaConfig`: the base `VecchiaConfig` object specifying the data you have and the conditioning sets for each measurement.

- `pred_pts`: a vector of points (either `Vector{SVector{D,Float64}}` or `Vector{Float64}`) to predict at.

- `ncondition`: the number of knn points to condition on for each prediction location.

OPTIONAL KEYWORD ARGUMENTS:

- `fixed_use_indices=nothing`: all with `ncondition` many nearest neighbors, you can provide additional indices to add to each conditioning set for the prediction points.

- `separable=false`: an indicator flag to determine whether or not to allow one prediction point to be in the conditioning set of another. If you do allow that, then you can only predict the unknown values jointly, and it requires assembling the reverse Cholesky factor. This is O(n) work and scalable, but has a higher prefactor than the alternative. Depending on your problem, `separable=true` may work nearly as well and be much faster.
"""
struct PredictionConfig{D,F}
  kernel::F
  pts::Vector{SVector{D,Float64}}
  data::Matrix{Float64}
  condix::Vector{Vector{Int64}}
  separable::Bool
end

function PredictionConfig(cfg::VecchiaConfig{H,D,F}, pred_pts::Vector{SVector{D,Float64}},
                          ncondition; fixed_use_indices=nothing, separable=false) where{H,D,F}
  check_singleton_sets(cfg)
  n      = length(cfg.condix)
  jpts   = vcat(reduce(vcat, cfg.pts), pred_pts)
  data   = reduce(vcat, cfg.data)
  condix = copy(cfg.condix)
  if fixed_use_indices isa AbstractVector{Int64}
    fixed_use_indices = fill(fixed_use_indices, length(pred_pts))
  end
  tree    = HierarchicalNSW(jpts)
  add_to_graph!(tree, 1:n)
  for j in eachindex(pred_pts)
    knns = Int64.(knn_search(tree, pred_pts[j], ncondition)[1])
    if !isnothing(fixed_use_indices)
      knns = sort(unique(vcat(fixed_use_indices[j], knns)))
    end
    push!(condix, knns)
    (!separable && (j < length(pred_pts))) && add_to_graph!(tree, n+j)
  end
  PredictionConfig(cfg.kernel, jpts, data, condix, separable)
end

function PredictionConfig(cfg::VecchiaConfig{H,1,F}, pred_pts::Vector{Float64},
                          ncondition; fixed_use_indices=nothing, separable=false) where{H,F}
  PredictionConfig(cfg, [SA[x] for x in pred_pts], ncondition;
                   fixed_use_indices=fixed_use_indices, separable=separable)
end

function jointconfig(pcfg::PredictionConfig{D,F}) where{D,F}
  (n, m) = (size(pcfg.data, 1), length(pcfg.pts) - size(pcfg.data, 1))
  _data  = hcat.(eachrow(pcfg.data))
  _pts   = [[x] for x in pcfg.pts]
  jcfg   = VecchiaConfig(pcfg.kernel, _data, _pts, pcfg.condix)
  (n, m, jcfg)
end

function _joint_knnpredict(pcfg::PredictionConfig{D,F}, params) where {D,F}
  (n, m, jcfg) = jointconfig(pcfg)
  Us     = UpperTriangular(sparse(rchol(jcfg, params; issue_warning=false)))
  em     = zeros(m+n, m)
  foreach(j->(em[n+j,j] = 1.0), 1:m)
  cols   = adjoint(Us)\(Us\em)
  pred_pts_marginal = cols[(n+1):(n+m), :]
  pred_pts_cross    = cols[1:n, :]
  Usnn = UpperTriangular(Us[1:n, 1:n])
  solved_cross = Usnn*(Usnn'*pred_pts_cross) 
  solved_cross'*reduce(vcat, pcfg.data)
end

function _separable_knnpredict(pcfg::PredictionConfig{D,F}, params) where{D,F}
  (n, m)        = (size(pcfg.data, 1), length(pcfg.pts) - size(pcfg.data, 1))
  pcondix       = view(pcfg.condix, (n+1):(n+m))
  pts           = view(pcfg.pts, 1:n)
  ppts          = view(pcfg.pts, (n+1):(n+m))
  szmax         = maximum(length, pcondix)
  _marginal_buf = zeros(szmax, szmax)
  _cross_buf    = zeros(szmax)
  out           = zeros(m, size(pcfg.data, 2))
  for j in eachindex(ppts)
    szj          = length(pcondix[j])
    marginal_buf = view(_marginal_buf, 1:szj, 1:szj)
    cross_buf    = view(_cross_buf, 1:szj)
    cpts         = view(pts, pcondix[j])
    for k in 1:szj
      cross_buf[k] = pcfg.kernel(cpts[k], ppts[j], params)
    end
    updatebuf!(marginal_buf, cpts, cpts, pcfg.kernel, params, skipltri=true)
    marginal_cov_f = cholesky!(Symmetric(marginal_buf))
    ldiv!(marginal_cov_f, cross_buf)
    for k in 1:size(pcfg.data, 2)
      out[j,k] = dot(cross_buf, view(pcfg.data, pcondix[j], k))
    end
  end
  out
end

function knnpredict(pcfg::PredictionConfig{D,F}, params) where{D,F}
  if pcfg.separable
    return _separable_knnpredict(pcfg, params)
  else
    return _joint_knnpredict(pcfg, params)
  end
end

function dense_posterior(pcfg::PredictionConfig{D,F}, params) where{D,F}
  pcfg.separable && @warn "'Separable' PredictionConfig objects may lead to unrealistic conditional simulations. Proceed with caution."
  (n, m, jcfg)  = jointconfig(pcfg)
  # for getting the necessary blocks of the actual covariance matrix, consider
  # the following. If Σ^{-1} = U U^T, then Σ v = U^{-T} (U^{-1} v). And to get
  # the Vecchia-implied marginal covariance of the points we are predicting at
  # and the cross covariance between those points and the observed data, we just
  # need to compute Σ [e_{n+1}, ..., e_{n+m}], where e_j is the coordinate
  # vector of zeros except e_j[j] = 1.0, n is the number of observed points, and
  # m is the number of prediction points.
  Us     = UpperTriangular(sparse(rchol(jcfg, params)))
  em     = zeros(m+n, m)
  foreach(j->(em[n+j,j] = 1.0), 1:m)
  cols   = adjoint(Us)\(Us\em)
  pred_pts_marginal = cols[(n+1):(n+m), :]
  pred_pts_cross    = cols[1:n, :]
  # finally, to get the inverse of the covariance of the data we have observed,
  # that can be applied with just the [1:n, 1:n] block of Us.
  Usnn = UpperTriangular(Us[1:n, 1:n])
  solved_cross = Usnn*(Usnn'*pred_pts_cross) # Σ_{data}^{-1} Σ_{cross}
  cond_mean    = solved_cross'*reduce(vcat, pcfg.data)
  cond_var     = pred_pts_marginal - solved_cross'pred_pts_cross
  (;cond_mean, cond_var)
end

# TODO (cg 2025/05/29 12:18): accept an RNG?
function cond_sim(pcfg::PredictionConfig{D,F}, params) where{D,F}
  pcfg.separable && @warn "'Separable' PredictionConfig objects may lead to unrealistic conditional simulations. Proceed with caution."
  (n, m, jcfg) = jointconfig(pcfg)
  if minimum(length, jcfg.condix[(n+1):end]) < 30
    @warn "For small numbers of conditioning points (<= 30 from slight anecdata), this method can give poor results. See issue #10 on github for more details." 
  end
  Us    = sparse(Vecchia.rchol(jcfg, params))
  Usnn  = Us[1:n, 1:n]
  # conditional simulation using standard tricks with Cholesky factors:
  data  = reduce(vcat, pcfg.data)
  rawwn = randn(m, size(data, 2))
  jwn   = vcat(Usnn'*data, rawwn)
  sims=(Us'\jwn)[(n+1):end, :]
end

