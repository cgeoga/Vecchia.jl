
# TODO (cg 2025/04/14 10:21): Would be very easy to make this parallel. 
function knnpredict(pts::Vector{SVector{D,Float64}}, data::Matrix{Float64}, kernel::K, 
                    params, pred_pts::Vector{SVector{D,Float64}}, ncondition) where{D,K}
  tree  = KDTree(pts)
  idxs  = knn(tree, pred_pts, ncondition)[1]
  pts_buf      = Vector{SVector{2,Float64}}(undef, ncondition)
  marginal_buf = zeros(ncondition, ncondition)
  cross_buf    = zeros(ncondition)
  out          = zeros(length(pred_pts), size(data, 2))
  for j in eachindex(pred_pts)
    cpts = view(pts, idxs[j])
    for k in 1:ncondition
      cross_buf[k] = kernel(cpts[k], pred_pts[j], params)
    end
    updatebuf!(marginal_buf, cpts, cpts, kernel, params, skipltri=true)
    marginal_cov_f = cholesky!(Symmetric(marginal_buf))
    ldiv!(marginal_cov_f, cross_buf)
    for k in 1:size(data, 2)
      out[j,k] = dot(cross_buf, view(data, idxs[j], k))
    end
  end
  out
end

function knnpredict(vc::VecchiaConfig{H,D,F}, params,
                    pred_pts::Vector{SVector{D,Float64}};
                    ncondition=maximum(length, vc.condix)) where{H,D,F}
  vpts  = reduce(vcat, vc.pts)
  vdata = reduce(vcat, vc.data)
  knnpredict(vpts, vdata, vc.kernel, params, pred_pts, ncondition)
end

function dense_posterior(vc::VecchiaConfig{H,D,F}, params,
                         pred_pts::Vector{SVector{D,Float64}};
                         ncondition=maximum(length, vc.condix)) where{H,D,F}
  # get the given data points and make a tree for fast conditioning set
  # collection.
  (n, m) = (sum(length, vc.pts), length(pred_pts))
  pts    = reduce(vcat, vc.pts)
  # create the new conditioning set elements for the joint configuration of
  # given and prediction points.
  jcondix = copy(vc.condix)
  sizehint!(jcondix, length(vc.condix) + length(pred_pts)) # could also pre-allocate
  for k in eachindex(pred_pts)
    tree       = KDTree(vcat(pts, pred_pts[1:(k-1)]))
    k_cond_ixs = NearestNeighbors.knn(tree, pred_pts[k], min(ncondition, n+(k-1)))[1]
    sort!(k_cond_ixs)
    push!(jcondix, k_cond_ixs)
  end
  # create the augmented/joint point list (for now, just singleton predictions):
  jpts  = vcat(vc.pts, [[x] for x in pred_pts])
  # create the final joint config object, not actually using any data.
  jcfg  = VecchiaConfig(vc.kernel, [fill(NaN, (1,1)) for _ in 1:length(jpts)], jpts, jcondix)
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
  cond_mean    = solved_cross'*reduce(vcat, vc.data)
  cond_var     = pred_pts_marginal - solved_cross'pred_pts_cross
  # return:
  (;cond_mean, cond_var)
end


function cond_sim(vc::Vecchia.VecchiaConfig{H,D,F}, params,
                  pred_pts::Vector{SVector{D,Float64}};
                  ncondition=maximum(length, vc.condix)) where{H,D,F}
  if ncondition < 30
    @warn "For small numbers of conditioning points (<= 30 from slight anecdata), this method can give poor results. See issue #10 on github for more details." 
  end
  # get the given data points and make a tree for fast conditioning set
  # collection.
  (n, m) = (sum(length, vc.pts), length(pred_pts))
  pts    = reduce(vcat, vc.pts)
  # create the new conditioning set elements for the joint configuration of
  # given and prediction points.
  jcondix = copy(vc.condix)
  sizehint!(jcondix, length(vc.condix) + length(pred_pts)) # could also pre-allocate
  # TODO (cg 2024/12/27 10:17): I really would like to switch to a dynamic tree
  # object for kNN queries. I expect that that is the clear bottleneck here.
  for k in eachindex(pred_pts)
    tree       = KDTree(vcat(pts, pred_pts[1:(k-1)]))
    k_cond_ixs = NearestNeighbors.knn(tree, pred_pts[k], min(ncondition, n+(k-1)))[1]
    sort!(k_cond_ixs)
    push!(jcondix, k_cond_ixs)
  end
  # create the augmented/joint point list (for now, just singleton predictions):
  jpts  = vcat(vc.pts, [[x] for x in pred_pts])
  # create the final joint config object. The data being passed in here isn't a
  # compliant size, but we'll never touch it.
  jcfg  = Vecchia.VecchiaConfig(vc.kernel, [hcat(NaN) for _ in eachindex(jpts)], 
                                jpts, jcondix)
  Us    = sparse(Vecchia.rchol(jcfg, params))
  Usnn  = Us[1:n, 1:n]
  # conditional simulation using standard tricks with Cholesky factors:
  data  = reduce(vcat, vc.data)
  rawwn = randn(length(pred_pts), size(data, 2))
  jwn   = vcat(Usnn'*data, rawwn)
  sims=(Us'\jwn)[(n+1):end, :]
end

